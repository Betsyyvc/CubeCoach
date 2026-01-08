"""Wrapper around python-kociemba to compute solution strings."""
try:
    import kociemba
except Exception:
    kociemba = None


class KociembaSolver:
    def __init__(self):
        if kociemba is None:
            raise RuntimeError("kociemba package not available. Install `kociemba`.")

    def solve(self, facelets):
        """facelets: 54-char string in facelet order expected by kociemba."""
        return kociemba.solve(facelets)
