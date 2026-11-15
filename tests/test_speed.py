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
