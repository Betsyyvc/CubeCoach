import pytest

try:
    import kociemba
except Exception:
    kociemba = None

from cubecoach.solver.kociemba_solver import KociembaSolver


@pytest.mark.skipif(kociemba is None, reason="kociemba not installed")
def test_solved_cube_returns_string():
    solver = KociembaSolver()
    # Facelet string for a solved cube (U R F D L B faces)
    facelets = 'UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB'
    sol = solver.solve(facelets)
    assert isinstance(sol, str)
