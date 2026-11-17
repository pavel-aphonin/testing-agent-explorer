"""Speed (PER-899): no work a step does not need."""

from __future__ import annotations

from dataclasses import dataclass

from explorer.engine import ExplorationEngine
from explorer.modes import ExplorationMode
from tests.test_engine import _app, _button, _PopupController


@dataclass
class _CountingShots(_PopupController):
    shots: int = 0

    async def take_screenshot(self) -> str:
        self.shots += 1
        return "<png>"


def _engine(tmp_path, controller):
    return ExplorationEngine(controller=controller, app_bundle_id="com.markov.testbed",
                             output_dir=str(tmp_path), mode=ExplorationMode.MC)


async def test_a_known_screen_is_not_photographed_again(tmp_path):
    controller = _CountingShots(tree=_app("Testbed", _button("Профиль", 200), _button("Счета", 260)))
    engine = _engine(tmp_path, controller)

    first = await engine._capture_screen()
    engine.graph.add_node(first)
    again = await engine._capture_screen()

    assert first.screenshot_b64 == "<png>"
    assert again.screen_id == first.screen_id
    assert controller.shots == 1


async def test_every_new_screen_gets_a_screenshot(tmp_path):
    controller = _CountingShots(tree=_app("Testbed", _button("Профиль", 200)))
    engine = _engine(tmp_path, controller)
    engine.graph.add_node(await engine._capture_screen())
    controller.tree = _app("Testbed", _button("Назад", 60), _button("Редактировать", 300))

    other = await engine._capture_screen()

    assert other.screenshot_b64 == "<png>"
    assert controller.shots == 2


@dataclass
class _FlakyTaps(_CountingShots):
    failures: int = 0
    taps: list = None

    async def tap_at(self, x, y):
        from tests.test_engine import _Result

        self.taps = (self.taps or []) + [(x, y)]
        if self.failures:
            self.failures -= 1
            return _Result(error="Timed out creating the simulator remote automation session")
        return _Result()


async def _tap_once(tmp_path, monkeypatch, failures):
    from tests.test_engine import _no_sleep

    monkeypatch.setattr("explorer.engine.asyncio.sleep", _no_sleep)
    controller = _FlakyTaps(tree=_app("Testbed", _button("Профиль", 200)), failures=failures)
    engine = _engine(tmp_path, controller)
    node = await engine._capture_screen()
    engine.graph.add_node(node)
    engine.current_screen_id = node.screen_id
    element = node.interactive_elements[0]
    await engine._explore_tap(element)
    return engine, node, element, controller


async def test_one_undelivered_tap_is_retried(tmp_path, monkeypatch):
    engine, node, _, controller = await _tap_once(tmp_path, monkeypatch, failures=1)
    assert len(controller.taps) == 2


async def test_an_undelivered_tap_leaves_the_element_untried(tmp_path, monkeypatch):
    engine, node, element, _ = await _tap_once(tmp_path, monkeypatch, failures=2)
    assert element.label in {el.label for el in engine.graph.get_unexplored_actions(node.screen_id)}


async def test_an_element_whose_taps_keep_failing_is_given_up(tmp_path, monkeypatch):
    engine, node, element, controller = await _tap_once(tmp_path, monkeypatch, failures=4)
    await engine._explore_tap(element)
    assert engine.graph.get_unexplored_actions(node.screen_id) == []


@dataclass
class _Scripted(_CountingShots):
    """Shows ``screens[i]`` on the i-th read, then the last one."""

    screens: list = None
    reads: int = 0

    async def get_ui_elements(self):
        tree = self.screens[min(self.reads, len(self.screens) - 1)]
        self.reads += 1
        return tree


def _clocked(monkeypatch):
    clock = {"t": 0.0}

    async def sleep(seconds):
        clock["t"] += seconds

    monkeypatch.setattr("explorer.engine.asyncio.sleep", sleep)
    monkeypatch.setattr("explorer.engine._clock", lambda: clock["t"])
    return clock


HOME = _app("Testbed", _button("Профиль", 200), _button("Счета", 260))
PROFILE = _app("Testbed", _button("Назад", 60), _button("Редактировать", 300))


async def _outcome(tmp_path, monkeypatch, screens):
    clock = _clocked(monkeypatch)
    controller = _Scripted(tree=HOME, screens=[HOME])
    engine = _engine(tmp_path, controller)
    home = await engine._capture_screen()
    engine.graph.add_node(home)
    controller.screens, controller.reads = screens, 0
    node = await engine._await_outcome(home.screen_id)
    return engine, home, node, clock, controller


async def test_a_fast_transition_is_taken_within_a_second(tmp_path, monkeypatch):
    engine, home, node, clock, _ = await _outcome(tmp_path, monkeypatch, [PROFILE])
    assert node.screen_id != home.screen_id
    assert clock["t"] < 1.0


async def test_a_tap_without_effect_waits_the_full_window(tmp_path, monkeypatch):
    from explorer.engine import OUTCOME_MAX_S

    engine, home, node, clock, _ = await _outcome(tmp_path, monkeypatch, [HOME])
    assert node.screen_id == home.screen_id
    assert clock["t"] >= OUTCOME_MAX_S


async def test_a_late_transition_inside_the_window_is_caught(tmp_path, monkeypatch):
    # «Открыть настройки iOS»: nothing for ~2 s, then another screen.
    engine, home, node, clock, _ = await _outcome(tmp_path, monkeypatch, [HOME] * 7 + [PROFILE])
    assert node.screen_id != home.screen_id


async def test_a_screen_still_loading_is_read_until_it_holds(tmp_path, monkeypatch):
    loaded = _app("Testbed", _button("Назад", 60), _button("Редактировать", 300), _button("Удалить", 360))
    engine, home, node, clock, _ = await _outcome(tmp_path, monkeypatch, [PROFILE, loaded])
    assert {el.label for el in node.interactive_elements} >= {"Удалить"}


async def test_a_new_sparse_screen_gets_time_to_load(tmp_path, monkeypatch):
    from explorer.engine import SPARSE_SCREEN_RECHECK_DELAY_S

    sparse = _app("Testbed", _button("Назад", 60))
    engine, home, node, clock, _ = await _outcome(tmp_path, monkeypatch, [sparse])
    assert clock["t"] >= SPARSE_SCREEN_RECHECK_DELAY_S


async def test_a_screen_is_accepted_only_once_it_holds(tmp_path, monkeypatch):
    more = _app("Testbed", _button("Назад", 60), _button("Редактировать", 300), _button("Удалить", 360))
    most = _app("Testbed", _button("Назад", 60), _button("Редактировать", 300), _button("Удалить", 360),
                _button("Избранное", 420))
    engine, home, node, clock, _ = await _outcome(tmp_path, monkeypatch, [PROFILE, more, most])
    assert "Избранное" in {el.label for el in node.interactive_elements}


async def test_a_variant_of_the_same_screen_is_no_change(tmp_path, monkeypatch):
    from explorer.engine import OUTCOME_MAX_S

    cards = [_button(f"Карта {c}", 100 + 40 * i) for i, c in enumerate("абвгде")]
    clock = _clocked(monkeypatch)
    controller = _Scripted(tree=_app("Деньги", *cards), screens=[_app("Деньги", *cards)])
    engine = _engine(tmp_path, controller)
    main = await engine._capture_screen()
    engine.graph.add_node(main)
    controller.screens = [_app("Деньги", *cards, _button("Карта ж", 400))]
    controller.reads = 0

    await engine._await_outcome(main.screen_id)

    assert clock["t"] >= OUTCOME_MAX_S


async def test_the_pin_is_not_typed_again_while_the_app_checks_it(tmp_path, monkeypatch):
    # aweassist run 10: a spinner over the keypad while the server checked
    # the code; the agent typed the PIN twice more and spent its budget.
    from tests.test_engine import _KeypadController, _keypad_engine, _keypad_tree, _no_sleep

    monkeypatch.setattr("explorer.engine.asyncio.sleep", _no_sleep)

    class _SlowServer(_KeypadController):
        checks = 0

        async def get_ui_elements(self):
            if len(self.typed) == 4 and self.checks < 5:
                self.checks += 1
                checking = _keypad_tree()
                checking[0]["children"] = []
                checking.append({"type": "GenericElement", "label": "Числовой код", "value": "Введено символов 4",
                                 "test_id": "", "enabled": True,
                                 "frame": {"x": 100, "y": 250, "width": 200, "height": 20}, "children": []})
                return checking
            return self.tree

        async def tap_by_id(self, test_id):
            self.taps_by_id.append(test_id)
            self.typed += test_id
            if len(self.typed) == 4 and self.typed == self.pin:
                self.tree = _app("Деньги", _button("Главная", 200))
            from tests.test_engine import _Result
            return _Result()

    controller = _SlowServer()
    engine = _keypad_engine(tmp_path, controller, test_data={"pin": "1234"})
    node = await engine._capture_screen()
    engine.graph.add_node(node)
    engine.current_screen_id = node.screen_id

    assert await engine._enter_pin_here(step=1)

    assert controller.taps_by_id == list("1234")
    assert engine._pin_failures == 0
    assert [el.label for el in engine.graph.nodes[engine.current_screen_id].interactive_elements] == ["Главная"]


async def test_a_keypad_that_comes_back_changed_is_still_a_rejection(tmp_path, monkeypatch):
    from tests.test_engine import _KeypadController, _keypad_engine, _keypad_tree, _no_sleep

    monkeypatch.setattr("explorer.engine.asyncio.sleep", _no_sleep)
    controller = _KeypadController()
    engine = _keypad_engine(tmp_path, controller, test_data={"pin": "9999"})
    node = await engine._capture_screen()
    engine.graph.add_node(node)
    engine.current_screen_id = node.screen_id
    rejected = _keypad_tree()
    rejected.append({"type": "Heading", "label": "Неверный код", "value": None, "test_id": "", "enabled": True,
                     "frame": {"x": 100, "y": 150, "width": 200, "height": 30}, "children": []})
    controller.tree = rejected

    await engine._enter_pin_here(step=1)

    assert engine.current_screen_id != node.screen_id
    assert engine._pin_failures == 1


def _reading_client(monkeypatch):
    import json as _json

    from explorer import axe_client
    from explorer.axe_client import AXeExplorerClient

    reads = []
    tree = _json.dumps([{"type": "Application", "AXLabel": "Testbed", "frame": {}, "children": []}]).encode()

    class _Proc:
        def __init__(self, out):
            self.returncode, self.out = 0, out

        async def communicate(self):
            return self.out, b""

    async def fake_exec(*args, **kwargs):
        if args[1] == "describe-ui":
            reads.append(1)
            return _Proc(tree)
        return _Proc(b"")

    async def no_hid(args):
        pass

    monkeypatch.setattr(axe_client.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(axe_client, "_note_hid_connection", no_hid)
    client = AXeExplorerClient()
    client._udid, client._width, client._height = "SIM", 402, 874
    return client, reads


async def test_a_tree_just_read_is_reused_until_the_next_input(monkeypatch):
    from explorer import axe_client

    client, reads = _reading_client(monkeypatch)
    await client.get_ui_elements()
    await client.get_ui_elements(reuse=True)
    assert len(reads) == 1

    await axe_client._run([axe_client.AXE, "tap", "-x", "1", "-y", "1", "--udid", "SIM"])
    await client.get_ui_elements(reuse=True)
    assert len(reads) == 2


async def test_polls_always_read_afresh(monkeypatch):
    client, reads = _reading_client(monkeypatch)
    await client.get_ui_elements()
    await client.get_ui_elements()
    assert len(reads) == 2


async def test_an_old_read_is_not_reused(monkeypatch):
    from explorer import axe_client

    client, reads = _reading_client(monkeypatch)
    await client.get_ui_elements()
    real = axe_client.time.monotonic
    monkeypatch.setattr(axe_client.time, "monotonic", lambda: real() + axe_client.TREE_REUSE_S + 1)
    await client.get_ui_elements(reuse=True)
    assert len(reads) == 2


def test_web_elements_do_not_count_in_the_native_fingerprint():
    from explorer.engine import native_fingerprint

    native = HOME
    with_web = HOME + [{"type": "Link", "label": "Открыть деталь", "web": True,
                        "frame": {"x": 16, "y": 227, "width": 146, "height": 26}}]
    assert native_fingerprint(native) == native_fingerprint(with_web)


@dataclass
class _AxeLike(_CountingShots):
    """Records which reads probe web pages; shows ``screens`` in order."""

    screens: list = None
    reads: list = None
    scan_web_content: bool = True

    async def get_ui_elements(self, reuse=False, web=True):
        self.reads = (self.reads or []) + [web]
        index = min(len(self.reads) - 1, len(self.screens) - 1)
        return self.screens[index]


async def test_polls_skip_web_pages_and_the_screen_is_read_in_full_once(tmp_path, monkeypatch):
    from explorer.engine import native_fingerprint

    _clocked(monkeypatch)
    controller = _AxeLike(tree=HOME, screens=[PROFILE])
    engine = _engine(tmp_path, controller)
    engine.graph.add_node(await engine._capture_screen())
    controller.reads = []

    node = await engine._await_outcome("home-id", native_fingerprint(HOME))

    assert {el.label for el in node.interactive_elements} == {"Назад", "Редактировать"}
    assert controller.reads.count(True) == 1          # the final capture
    assert controller.reads[:-1] and not any(controller.reads[:-1])  # polls: native only


async def test_a_native_poll_still_waits_out_a_tap_without_effect(tmp_path, monkeypatch):
    from explorer.engine import OUTCOME_MAX_S, native_fingerprint

    clock = _clocked(monkeypatch)
    controller = _AxeLike(tree=HOME, screens=[HOME])
    engine = _engine(tmp_path, controller)
    home = await engine._capture_screen()
    engine.graph.add_node(home)

    node = await engine._await_outcome(home.screen_id, native_fingerprint(HOME))

    assert node.screen_id == home.screen_id
    assert clock["t"] >= OUTCOME_MAX_S


async def test_a_native_poll_waits_for_a_changed_screen_to_hold(tmp_path, monkeypatch):
    from explorer.engine import native_fingerprint

    _clocked(monkeypatch)
    more = _app("Testbed", _button("Назад", 60), _button("Редактировать", 300), _button("Удалить", 360))
    most = _app("Testbed", _button("Назад", 60), _button("Редактировать", 300), _button("Удалить", 360),
                _button("Избранное", 420))
    controller = _AxeLike(tree=HOME, screens=[PROFILE, more, most, most])
    engine = _engine(tmp_path, controller)
    engine.graph.add_node(await engine._capture_screen())
    controller.reads = []

    node = await engine._await_outcome("home-id", native_fingerprint(HOME))

    assert "Избранное" in {el.label for el in node.interactive_elements}
    assert controller.reads[:4] == [False, False, False, False]  # read until two polls agree


def _web_client(monkeypatch):
    import json as _json

    from explorer import axe_client
    from explorer.axe_client import AXeExplorerClient

    calls = {"tree": 0, "point": 0}
    tree = _json.dumps([{"type": "Application", "AXLabel": "Testbed", "frame": {"x": 0, "y": 0, "width": 402, "height": 874},
        "children": [{"type": "ScrollArea", "AXLabel": "", "frame": {"x": 0, "y": 154, "width": 402, "height": 686},
                      "children": []}]}]).encode()
    link = _json.dumps({"type": "Link", "AXLabel": "Открыть деталь",
                        "frame": {"x": 16, "y": 227, "width": 146, "height": 26}}).encode()

    class _Proc:
        def __init__(self, out):
            self.returncode, self.out = 0, out

        async def communicate(self):
            return self.out, b""

    async def fake_exec(*args, **kwargs):
        if "--point" in args:
            calls["point"] += 1
            return _Proc(link)
        calls["tree"] += 1
        return _Proc(tree)

    monkeypatch.setattr(axe_client.asyncio, "create_subprocess_exec", fake_exec)
    client = AXeExplorerClient()
    client._udid, client._width, client._height = "SIM", 402, 874
    return client, calls


async def test_a_native_read_does_not_probe_web_pages(monkeypatch):
    client, calls = _web_client(monkeypatch)
    elements = await client.get_ui_elements(web=False)
    assert calls["point"] == 0
    assert "Открыть деталь" not in [e["label"] for e in elements]


async def test_a_native_read_is_not_reused_for_a_full_one(monkeypatch):
    client, calls = _web_client(monkeypatch)
    await client.get_ui_elements(web=False)
    elements = await client.get_ui_elements(reuse=True)
    assert calls["tree"] == 2
    assert "Открыть деталь" in [e["label"] for e in elements]


async def test_the_first_screen_is_a_launch_screen(tmp_path):
    controller = _CountingShots(tree=HOME)
    engine = _engine(tmp_path, controller)
    engine.max_steps = 0

    graph = await engine.run()

    assert graph.launch_screens == [next(iter(graph.nodes))]


async def test_a_diagnostic_keeps_the_tree_and_a_screenshot(tmp_path):
    import base64
    import json as _json

    from explorer.models import ActionDetail, ActionType, GraphEdge

    controller = _CountingShots(tree=HOME)
    controller._last_read = (0, 0.0, [{"type": "Button", "label": "Places"}], True)

    async def png():
        return base64.b64encode(b"PNG").decode()

    controller.take_screenshot = png
    engine = _engine(tmp_path, controller)
    edge = GraphEdge(source_screen_id="a", target_screen_id="b",
                     action=ActionDetail(action_type=ActionType.TAP, target_label="Places"))

    await engine._diagnose("replay-failed", edge)

    folder = tmp_path / "diagnostics"
    record = _json.loads((folder / "001-replay-failed.json").read_text())
    assert record["tree"] == [{"type": "Button", "label": "Places"}]
    assert record["edge"]["action"]["target_label"] == "Places"
    assert (folder / "001-replay-failed.png").read_bytes() == b"PNG"


async def test_a_tap_without_effect_is_read_only_a_few_times(tmp_path, monkeypatch):
    from explorer.engine import native_fingerprint

    _clocked(monkeypatch)
    controller = _AxeLike(tree=HOME, screens=[HOME])
    engine = _engine(tmp_path, controller)
    home = await engine._capture_screen()
    engine.graph.add_node(home)
    controller.reads = []

    await engine._await_outcome(home.screen_id, native_fingerprint(HOME))

    assert len(controller.reads) <= 7  # was 12: a read every 0.3 s for 3.5 s


async def test_an_element_that_cannot_be_brought_on_screen_leaves_a_diagnostic(tmp_path):
    """wikipedia-6: 71 taps on «keys» below the screen failed without a trace
    of what the tree held; the tree and a screenshot are kept now, a few times."""
    import json as _json

    controller = _CountingShots(tree=HOME)
    controller._last_read = (0, 0.0, [{"type": "Button", "label": "z"}], True)
    engine = _engine(tmp_path, controller)
    element = _snapshot("z")

    for _ in range(9):
        await engine._diagnose_unreachable(element)

    files = sorted(p.name for p in (tmp_path / "diagnostics").glob("*.json"))
    assert files == [f"{n:03d}-not-on-screen.json" for n in range(1, 6)]
    record = _json.loads((tmp_path / "diagnostics" / files[0]).read_text())
    assert record["element"]["label"] == "z" and record["tree"] == [{"type": "Button", "label": "z"}]


async def test_unreachable_diagnostics_do_not_use_up_the_replay_ones(tmp_path):
    controller = _CountingShots(tree=HOME)
    engine = _engine(tmp_path, controller)
    for _ in range(9):
        await engine._diagnose_unreachable(_snapshot("z"))
    assert engine._diagnostics == 0


def _snapshot(label):
    from explorer.models import ElementSnapshot

    return ElementSnapshot(kind="button", element_type="Button", label=label, test_id=label,
                           frame={"x": 5, "y": 898, "width": 40, "height": 54})
