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
