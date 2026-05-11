"""Tiny safe expression evaluator for scenario edge conditions (PER-83).

Why not ``eval`` / ``ast.literal_eval``? We need comparisons + boolean
combinations + access to nested context like ``{{test_data.balance}}``,
which neither covers safely. Building a minimal recursive-descent
parser is ~150 lines and lets us reject anything that isn't on the
allowed grammar (no attribute access on arbitrary objects, no calls
into the standard library, etc).

Supported grammar (informal)::

    expr        := or_expr
    or_expr     := and_expr ('||' and_expr)*
    and_expr    := not_expr ('&&' not_expr)*
    not_expr    := '!' not_expr | comparison
    comparison  := primary (('==' | '!=' | '<' | '>' | '<=' | '>=') primary)?
    primary     := literal | variable | call | '(' expr ')'
    literal     := number | string | 'true' | 'false' | 'null'
    variable    := '{{' identifier ('.' identifier)* '}}'
    call        := identifier '(' (expr (',' expr)*)? ')'
    identifier  := [A-Za-z_][A-Za-z0-9_]*

Allowed function names: ``contains``, ``starts_with``, ``ends_with``,
``length``, ``lower``, ``upper``. Anything else raises a parse error.

Evaluation context is a regular dict; ``{{a.b.c}}`` looks up
``ctx["a"]["b"]["c"]``. Missing keys yield ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ExprError(Exception):
    """Raised on parse or evaluation errors. ``str(exc)`` is safe to
    surface to the user via the run log."""


# ─────────────────────────────────────── lexer

_TOKEN_TYPES = (
    "NUMBER", "STRING", "IDENT", "VAR",
    "LPAREN", "RPAREN", "COMMA",
    "EQ", "NEQ", "LT", "GT", "LE", "GE",
    "AND", "OR", "NOT",
    "TRUE", "FALSE", "NULL",
    "EOF",
)


@dataclass
class Token:
    kind: str
    value: Any
    pos: int


def _tokenize(src: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    n = len(src)
    while i < n:
        ch = src[i]
        if ch.isspace():
            i += 1
            continue
        # ``{{ ... }}`` variable reference. We capture the entire path
        # as one VAR token so the parser doesn't have to deal with the
        # ``.`` inside.
        if ch == "{" and i + 1 < n and src[i + 1] == "{":
            j = i + 2
            while j < n and not (src[j] == "}" and j + 1 < n and src[j + 1] == "}"):
                j += 1
            if j >= n:
                raise ExprError(f"unterminated variable reference at position {i}")
            path = src[i + 2:j].strip()
            if not path:
                raise ExprError(f"empty variable reference at position {i}")
            tokens.append(Token("VAR", path, i))
            i = j + 2
            continue
        # String literal — single or double quotes. Backslash is a
        # LITERAL character, not an escape: ``'a\\b'`` lexes to the
        # three-char string ``a\b``. This is intentional — scenario
        # condition expressions don't need string-internal escapes
        # (use ``{{var}}`` interpolation for dynamic content), and
        # the simpler lexer makes the rule easier to remember.
        if ch in ("'", '"'):
            quote = ch
            j = i + 1
            while j < n and src[j] != quote:
                j += 1
            if j >= n:
                raise ExprError(f"unterminated string literal at position {i}")
            tokens.append(Token("STRING", src[i + 1:j], i))
            i = j + 1
            continue
        # Number literal — integer or float.
        if ch.isdigit() or (ch == "-" and i + 1 < n and src[i + 1].isdigit()):
            j = i + 1
            while j < n and (src[j].isdigit() or src[j] == "."):
                j += 1
            raw = src[i:j]
            try:
                value: Any = float(raw) if "." in raw else int(raw)
            except ValueError as exc:
                raise ExprError(f"bad number literal {raw!r}") from exc
            tokens.append(Token("NUMBER", value, i))
            i = j
            continue
        # Identifier or keyword.
        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            word = src[i:j]
            kind = {
                "true": "TRUE", "false": "FALSE", "null": "NULL",
            }.get(word, "IDENT")
            tokens.append(Token(kind, word, i))
            i = j
            continue
        # Two-character operators first.
        two = src[i:i + 2]
        if two in ("==", "!=", "<=", ">=", "&&", "||"):
            kind = {
                "==": "EQ", "!=": "NEQ", "<=": "LE", ">=": "GE",
                "&&": "AND", "||": "OR",
            }[two]
            tokens.append(Token(kind, two, i))
            i += 2
            continue
        # Single-character.
        single = {
            "(": "LPAREN", ")": "RPAREN", ",": "COMMA",
            "<": "LT", ">": "GT", "!": "NOT",
        }.get(ch)
        if single is not None:
            tokens.append(Token(single, ch, i))
            i += 1
            continue
        raise ExprError(f"unexpected character {ch!r} at position {i}")
    tokens.append(Token("EOF", "", n))
    return tokens


# ─────────────────────────────────────── parser + evaluator (combined)


class _Parser:
    """Recursive-descent parser that evaluates as it goes — there's no
    point materialising an AST when the grammar is this small."""

    _ALLOWED_CALLS = {"contains", "starts_with", "ends_with", "length", "lower", "upper"}

    def __init__(self, tokens: list[Token], context: dict[str, Any]) -> None:
        self.tokens = tokens
        self.pos = 0
        self.context = context

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _eat(self, kind: str) -> Token:
        tok = self._peek()
        if tok.kind != kind:
            raise ExprError(
                f"expected {kind} at position {tok.pos}, got {tok.kind} ({tok.value!r})"
            )
        self.pos += 1
        return tok

    # expr := or
    def parse(self) -> Any:
        value = self._or()
        if self._peek().kind != "EOF":
            tok = self._peek()
            raise ExprError(f"unexpected token {tok.kind} ({tok.value!r}) at position {tok.pos}")
        return value

    def _or(self) -> Any:
        left = self._and()
        while self._peek().kind == "OR":
            self._eat("OR")
            right = self._and()
            left = bool(left) or bool(right)
        return left

    def _and(self) -> Any:
        left = self._not()
        while self._peek().kind == "AND":
            self._eat("AND")
            right = self._not()
            left = bool(left) and bool(right)
        return left

    def _not(self) -> Any:
        if self._peek().kind == "NOT":
            self._eat("NOT")
            return not bool(self._not())
        return self._comparison()

    def _comparison(self) -> Any:
        left = self._primary()
        kind = self._peek().kind
        if kind in ("EQ", "NEQ", "LT", "GT", "LE", "GE"):
            self._eat(kind)
            right = self._primary()
            return _compare(kind, left, right)
        return left

    def _primary(self) -> Any:
        tok = self._peek()
        if tok.kind == "NUMBER":
            self._eat("NUMBER")
            return tok.value
        if tok.kind == "STRING":
            self._eat("STRING")
            return tok.value
        if tok.kind == "TRUE":
            self._eat("TRUE")
            return True
        if tok.kind == "FALSE":
            self._eat("FALSE")
            return False
        if tok.kind == "NULL":
            self._eat("NULL")
            return None
        if tok.kind == "VAR":
            self._eat("VAR")
            return _resolve_var(self.context, tok.value)
        if tok.kind == "LPAREN":
            self._eat("LPAREN")
            value = self._or()
            self._eat("RPAREN")
            return value
        if tok.kind == "IDENT":
            name = tok.value
            self._eat("IDENT")
            self._eat("LPAREN")
            args: list[Any] = []
            if self._peek().kind != "RPAREN":
                args.append(self._or())
                while self._peek().kind == "COMMA":
                    self._eat("COMMA")
                    args.append(self._or())
            self._eat("RPAREN")
            if name not in self._ALLOWED_CALLS:
                raise ExprError(f"unknown function {name!r}")
            return _call(name, args)
        raise ExprError(f"unexpected token {tok.kind} ({tok.value!r}) at position {tok.pos}")


def _resolve_var(context: dict[str, Any], path: str) -> Any:
    parts = [p.strip() for p in path.split(".") if p.strip()]
    if not parts:
        return None
    cur: Any = context
    for p in parts:
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            return None
    return cur


def _compare(op: str, left: Any, right: Any) -> bool:
    # ``None`` is only equal to ``None``; mismatched types under
    # ordering ops collapse to False rather than raising — safer for
    # user-authored expressions.
    if op == "EQ":
        return left == right
    if op == "NEQ":
        return left != right
    if left is None or right is None:
        return False
    try:
        if op == "LT":
            return left < right
        if op == "GT":
            return left > right
        if op == "LE":
            return left <= right
        if op == "GE":
            return left >= right
    except TypeError:
        return False
    raise ExprError(f"unknown operator {op}")


def _call(name: str, args: list[Any]) -> Any:
    if name == "contains":
        if len(args) != 2:
            raise ExprError("contains() expects 2 arguments")
        haystack = args[0] or ""
        needle = args[1] or ""
        return str(needle) in str(haystack)
    if name == "starts_with":
        if len(args) != 2:
            raise ExprError("starts_with() expects 2 arguments")
        return str(args[0] or "").startswith(str(args[1] or ""))
    if name == "ends_with":
        if len(args) != 2:
            raise ExprError("ends_with() expects 2 arguments")
        return str(args[0] or "").endswith(str(args[1] or ""))
    if name == "length":
        if len(args) != 1:
            raise ExprError("length() expects 1 argument")
        v = args[0]
        if v is None:
            return 0
        try:
            return len(v)
        except TypeError:
            return 0
    if name == "lower":
        if len(args) != 1:
            raise ExprError("lower() expects 1 argument")
        return str(args[0] or "").lower()
    if name == "upper":
        if len(args) != 1:
            raise ExprError("upper() expects 1 argument")
        return str(args[0] or "").upper()
    raise ExprError(f"unknown function {name!r}")


def evaluate(expr: str, context: dict[str, Any]) -> bool:
    """Parse and evaluate ``expr`` against ``context``. Returns the
    truthiness of the result. Raises ``ExprError`` on syntax errors,
    unknown variables/functions, etc.

    Example::

        evaluate("{{test_data.role}} == \\"admin\\" || {{balance}} > 1000",
                 {"test_data": {"role": "admin"}, "balance": 500})
        # → True
    """
    tokens = _tokenize(expr)
    parser = _Parser(tokens, context)
    return bool(parser.parse())
