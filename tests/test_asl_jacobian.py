"""ASL (PyomoNLP) como evaluador del Jacobiano: mismos valores, 200x mas rapido.

El oraculo es `differentiate()` — la via que ya estaba en produccion. Si ASL
difiere de ella, ASL esta mal, no al reves.

El test que importa es el del ORDEN: el escritor .nl reordena variables y filas,
y en el GTAP 20x41 medimos 0/210,094 alineadas por casualidad. Usar ASL sin
reindexar da un Jacobiano completo pero PERMUTADO — converge a la raiz
equivocada en silencio. `test_order_mismatch_is_real` fija ese hecho.
"""
from __future__ import annotations

import pytest

pyomo = pytest.importorskip("pyomo")

from pyomo.environ import ConcreteModel, Constraint, Var, log, value  # noqa: E402
from pyomo.core.expr.calculus.derivatives import Modes, differentiate  # noqa: E402

from path_capi_python.pyomo_adapter import PyomoMCPAdapter  # noqa: E402


def _asl_available() -> bool:
    try:
        from pyomo.contrib.pynumero.asl import AmplInterface
        return bool(AmplInterface.available())
    except Exception:
        return False


requires_asl = pytest.mark.skipif(not _asl_available(), reason="pynumero ASL no disponible")


def _ces_model(n: int = 6) -> ConcreteModel:
    """Modelo cuadrado con la forma del GTAP: potencias fraccionarias y log."""
    m = ConcreteModel()
    m.I = range(n)
    m.x = Var(m.I, initialize=lambda _m, i: 1.0 + 0.1 * i, bounds=(0.01, None))

    def _rule(_m, i):
        a = _m.x[i] ** 0.7 * _m.x[(i + 1) % n] ** 0.3
        b = log(_m.x[(i + 2) % n])
        return a + b - (1.0 + 0.01 * i) == 0.0

    m.eq = Constraint(m.I, rule=_rule)
    return m


def _dense_jacobian_via_differentiate(cons, varlist):
    """Oraculo: J[i][j] via differentiate(), indexado por el orden dado."""
    from pyomo.core.expr.visitor import identify_variables

    pos = {id(v): j for j, v in enumerate(varlist)}
    J = [[0.0] * len(varlist) for _ in cons]
    for i, c in enumerate(cons):
        expr = c.body - value(c.lower)
        seen, vs = set(), []
        for v in identify_variables(expr, include_fixed=False):
            j = pos.get(id(v))
            if j is None or j in seen:
                continue
            seen.add(j)
            vs.append(v)
        if not vs:
            continue
        for v, d in zip(vs, differentiate(expr, wrt_list=vs, mode=Modes.reverse_numeric)):
            J[i][pos[id(v)]] = float(d)
    return J


@requires_asl
def test_asl_matches_differentiate_under_adapter_order():
    """Reindexado por identidad: ASL == differentiate, celda a celda."""
    m = _ces_model()
    adapter = PyomoMCPAdapter()
    cons = list(m.eq.values())
    # Orden del adaptador: DISTINTO del que elegira el escritor .nl
    varlist = list(reversed(list(m.x.values())))

    data = adapter.build_nonlinear_from_equality_constraints(
        m, constraints=cons, variables=varlist, jacobian_eval_mode="asl"
    )
    values = list(data.callback_jac([float(value(v)) for v in varlist]))

    # Reconstruir denso desde el formato por columnas de PATH
    st = data.jacobian_structure
    dense = [[0.0] * len(varlist) for _ in cons]
    k = 0
    for j in range(len(varlist)):
        for off in range(st.col_lengths[j]):
            row = st.row_indices[st.col_starts[j] - 1 + off] - 1
            dense[row][j] = values[k]
            k += 1

    expected = _dense_jacobian_via_differentiate(cons, varlist)
    for i in range(len(cons)):
        for j in range(len(varlist)):
            assert abs(dense[i][j] - expected[i][j]) < 1e-9, f"celda ({i},{j})"


@requires_asl
def test_order_mismatch_is_real():
    """El escritor .nl NO conserva el orden dado: por eso hay que reindexar."""
    from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
    from pyomo.common.modeling import unique_component_name
    from pyomo.environ import Objective

    m = _ces_model()
    varlist = list(reversed(list(m.x.values())))
    name = unique_component_name(m, "_probe_obj")
    m.add_component(name, Objective(expr=0.0))
    try:
        nlp = PyomoNLP(m)
        asl_order = [id(v) for v in nlp.get_pyomo_variables()]
    finally:
        m.del_component(name)

    assert set(asl_order) == {id(v) for v in varlist}, "mismo conjunto de variables"
    assert asl_order != [id(v) for v in varlist], (
        "si esto falla, el orden coincidio por casualidad y el test no prueba nada"
    )


@requires_asl
def test_asl_agrees_with_reverse_numeric_mode():
    """Las dos vias del adaptador deben coincidir sobre el MISMO modelo."""
    m = _ces_model()
    adapter = PyomoMCPAdapter()
    cons = list(m.eq.values())
    varlist = list(m.x.values())
    x = [float(value(v)) for v in varlist]

    a = adapter.build_nonlinear_from_equality_constraints(
        m, constraints=cons, variables=varlist, jacobian_eval_mode="reverse_numeric"
    )
    b = adapter.build_nonlinear_from_equality_constraints(
        m, constraints=cons, variables=varlist, jacobian_eval_mode="asl"
    )
    assert a.jacobian_structure.nnz == b.jacobian_structure.nnz
    assert list(a.jacobian_structure.row_indices) == list(b.jacobian_structure.row_indices)
    for u, v in zip(a.callback_jac(x), b.callback_jac(x)):
        assert abs(float(u) - float(v)) < 1e-9


@requires_asl
def test_asl_residual_matches_python_evaluation():
    """F(x) via ASL == F(x) via Pyomo's value(), under the adapter's own order."""
    m = _ces_model()
    adapter = PyomoMCPAdapter()
    cons = list(m.eq.values())
    varlist = list(reversed(list(m.x.values())))  # deliberately not the .nl order

    data = adapter.build_nonlinear_from_equality_constraints(
        m, constraints=cons, variables=varlist, jacobian_eval_mode="asl"
    )
    x = [float(value(v)) for v in varlist]
    got = list(data.callback_f(x))

    for v, xv in zip(varlist, x):
        v.set_value(xv, skip_validation=True)
    expected = [float(value(c.body - value(c.lower))) for c in cons]

    assert len(got) == len(expected)
    for i, (a, b) in enumerate(zip(got, expected)):
        assert abs(a - b) < 1e-9, f"row {i}: {a} vs {b}"


@requires_asl
def test_asl_residual_tracks_the_point():
    """A different x must give a different F — guards against a stale cache."""
    m = _ces_model()
    adapter = PyomoMCPAdapter()
    cons = list(m.eq.values())
    varlist = list(m.x.values())

    data = adapter.build_nonlinear_from_equality_constraints(
        m, constraints=cons, variables=varlist, jacobian_eval_mode="asl"
    )
    x0 = [float(value(v)) for v in varlist]
    f0 = list(data.callback_f(x0))
    x1 = [v * 1.05 for v in x0]
    f1 = list(data.callback_f(x1))
    assert any(abs(a - b) > 1e-9 for a, b in zip(f0, f1)), "F did not follow x"

    # and going back reproduces the original values
    f0b = list(data.callback_f(x0))
    for a, b in zip(f0, f0b):
        assert abs(a - b) < 1e-12


@requires_asl
def test_asl_residual_is_not_a_python_walk():
    """In "asl" mode F(x) must come from ASL, not from Pyomo's Python walker.

    Both give the same numbers, so correctness cannot tell them apart. What
    separates them is cost: the Python walk is O(expression tree) per row, ASL
    is one C call. On a model wide enough for the gap to clear timing noise,
    asl must be decisively faster than reverse_numeric — which walks the same
    trees the old residual did.

    Profiling one gtap7_20x41 solve put 29% of the time in that walk, against
    the Jacobian's 17%: after moving J to ASL, F became the larger half.
    """
    import time

    m = _ces_model(n=400)
    adapter = PyomoMCPAdapter()
    cons = list(m.eq.values())
    varlist = list(m.x.values())
    x = [float(value(v)) for v in varlist]

    slow = adapter.build_nonlinear_from_equality_constraints(
        m, constraints=cons, variables=varlist, jacobian_eval_mode="reverse_numeric"
    )
    fast = adapter.build_nonlinear_from_equality_constraints(
        m, constraints=cons, variables=varlist, jacobian_eval_mode="asl"
    )

    for a, b in zip(slow.callback_f(x), fast.callback_f(x)):
        assert abs(float(a) - float(b)) < 1e-9

    def best(fn, reps=5):
        fn(x)
        return min((lambda t0: (fn(x), time.perf_counter() - t0)[1])(time.perf_counter())
                   for _ in range(reps))

    t_slow = best(slow.callback_f)
    t_fast = best(fast.callback_f)
    assert t_fast * 3 < t_slow, (
        f"asl residual {t_fast:.5f}s vs python {t_slow:.5f}s — not the C path"
    )
