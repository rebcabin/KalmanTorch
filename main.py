import torch
import toolz
import numpy as np

"""See https://realpython.com/python-timer/"""

from dataclasses import dataclass, field
import time
from typing import Callable, Optional


class TimerError(Exception):
    """custom Exception for Timer errors"""


@dataclass
class Timer:
    """nestable"""
    text        : str = "Elapsed time: {:0.4f} seconds"
    logger      : Callable[[str], None] = print
    _start_time : Optional[float] = \
        field(default=None, init=False, repr=False)

    def start(self) -> None:
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        self.logger(self.text.format(elapsed_time))
        return elapsed_time

    # Protocol methods for context manager:
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc_info):
        self.stop()


def kalman_torch(b,  # # rows, cols, in Z; # rows in z
                 n,  # # rows, cols, in P; # rows in x
                 Z,  # b x b observation covariance
                 x,  # n x 1, current state
                 P,  # n x n, current covariance
                 A,  # b x n, current observation partials
                 z  # b x 1, current observation vector
                 ):
    """Recurrent Kalman filter for parameter estimation (no dynamics)."""

    # Transcribe the following sketch of Wolfram code (the intermediate
    # matrices are not necessary in Wolfram, but we need them in Python).
    #
    # noInverseKalman[Z_][{x_, P_}, {A_, z_}] :=
    #
    #   Module[{PAT, D, Res, DiRes, KRes, AP, DiAP, KAP},
    #
    #    1. PAT    = P.Transpose[A]         (* n x b *)
    #    2. D      = Z + A.PAT              (* b x b *)
    #    3. Res    = z - A.x                (* b x 1 *)
    #    4. DiRes  = LinearSolve[D, Res]    (* b x 1 *)
    #    5. KRes   = PAT.DiRes              (* n x 1 *)
    #    6. AP     = A.P                    (* n x 1 *)
    #    7. DiAP   = LinearSolve[D, AP]     (* b x n *)
    #    8. KAP    = PAT.DiAP               (* n x n *)
    #    9. x     <- x + KRes
    #   10. P     <- P - KAP
    #

    #
    #      PAT                P           AT
    #       b                 n           b
    #    / * * \         / * * * * \   / * * \
    #  n | * * |  <--  n | * * * * | n | * * |
    #    | * * |         | * * * * |   | * * |
    #    \ * * /         \ * * * * /   \ * * /

    pat = torch.matmul(P, torch.t(A))

    #
    #       D                 A          PAT           Z
    #       b                 n           b            b
    #  b / * * \  <--  b / * * * * \ n / * * \  + b / * * \
    #    \ * * /         \ * * * * /   | * * |      \ * * /
    #                                  | * * |
    #                                  \ * * /

    d = torch.add(torch.matmul(A, pat), Z)

    #            NOTA BENE
    #                |
    #     Res        |        A          x          z
    #      1         v        n          1          1
    #  b / * \  <--  - b / * * * * \ n / * \  + b / * \
    #    \ * /           \ * * * * /   | * |      \ * /
    #                                  | * |
    #                                  \ * /

    res = torch.sub(z, torch.matmul(A, x))

    #
    #    DiRes        Di = D^-1   Res
    #      1              b        1
    #  b / * \  <--  b / * * \ b / * \
    #    \ * /         \ * * /   \ * /

    di = torch.inverse(d)
    dires = torch.matmul(di, res)

    #
    #     KRes           PAT     DiRes
    #      1              b        1
    #  n / * \       n / * * \ b / * \
    #    | * |  <--    | * * |   \ * /
    #    | * |         | * * |
    #    \ * /         \ * * /

    kres = torch.matmul(pat, dires)

    #
    #         AP                  A             P
    #         n                   n             n
    #  b / * * * * \  <--  b / * * * * \ n / * * * * \
    #    \ * * * * /         \ * * * * /   | * * * * |
    #                                      | * * * * |
    #                                      \ * * * * /

    ap = torch.matmul(A, P)

    #
    #        DiAP           Di = D^-1       AP
    #         n                 b           n
    #  b / * * * * \  <--  b / * * \ b / * * * * \
    #    \ * * * * /         \ * * /   \ * * * * /

    diap = torch.matmul(di, ap)

    #
    #        KAP             PAT         DiAP
    #         n               b           n
    #  n / * * * * \  <--  / * * \ b / * * * * \
    #    | * * * * |     n | * * |   \ * * * * /
    #    | * * * * |       | * * |
    #    \ * * * * /       \ * * /

    kap = torch.matmul(pat, diap)

    #
    #      x            x          KRes
    #      1            1           1
    #  n / * \  <-- n / * \  + n / * \
    #    | * |        | * |      | * |
    #    | * |        | * |      | * |
    #    \ * /        \ * /      \ * /

    x = torch.add(x, kres)

    #                  NOTA BENE
    #                      |
    #         P            |       KAP               P
    #         n            v        n                n
    #  n / * * * * \  <--  - n / * * * * \  + n / * * * * \
    #    | * * * * |           | * * * * |      | * * * * |
    #    | * * * * |           | * * * * |      | * * * * |
    #    \ * * * * /           \ * * * * /      \ * * * * /

    p = torch.sub(P, kap)

    return x, p


def normal_equations():
    """Produces the estimate by linear regression without covariance
    (uncertainty)."""

    print("----------------------------------------------------------------")
    print("The Normal Equations for Linear Regression")

    x0 = torch.zeros(4)
    print({'x0': x0})
    a = torch.tensor([[1., 0., 0., 0.],
                      [1., 1., 1., 1.],
                      [1., -1., 1., -1.],
                      [1., -2., 4., -8.],
                      [1., 2., 4., 8.]])
    print({'A': a})
    zs = torch.tensor([-2.28442, -4.83168, -10.4601, 1.40488, -40.8079])
    print({'zs': zs})
    at = torch.t(a)
    print({'at': at})
    ata = torch.matmul(at, a)
    print({'ata': ata})
    atai = torch.inverse(torch.matmul(at, a))
    print({'atai': atai})
    atai_at = torch.matmul(atai, at)
    print({'atai_at': atai_at})
    atai_at_zs = torch.matmul(atai_at, zs)
    print({'expect': torch.tensor([-2.9751, 7.2700, -4.2104, -4.4558])})
    print({'atai_at_zs': atai_at_zs})


def kalman_torch_sample_by_hand():
    """Verify against equation 1 in https://vixra.org/pdf/1606.0328v1.pdf"""

    # print("----------------------------------------------------------------")
    # print("Explicit intermediate variables in a recurrence over five data.")

    x0 = torch.tensor([[x] for x in torch.zeros(4)])

    zs = torch.tensor([[z] for z in [-2.28442, -4.83168, -10.4601, 1.40488, -40.8079]])

    aas = torch.tensor([[a] for a in [[1., 0., 0., 0.],
                                      [1., 1., 1., 1.],
                                      [1., -1., 1., -1.],
                                      [1., -2., 4., -8.],
                                      [1., 2., 4., 8.]]])

    p0 = 1000. * torch.eye(4)

    Z = torch.tensor([[1.0]])

    x1, p1 = kalman_torch(1, 4, Z, x0, p0, aas[0], zs[0])
    # print({'x1': x1, 'p1': p1})

    x2, p2 = kalman_torch(1, 4, Z, x1, p1, aas[1], zs[1])
    # print({'x2': x2, 'p2': p2})

    x3, p3 = kalman_torch(1, 4, Z, x2, p2, aas[2], zs[2])
    # print({'x3': x3, 'p3': p3})

    x4, p4 = kalman_torch(1, 4, Z, x3, p3, aas[3], zs[3])
    # print({'x4': x4, 'p4': p4})

    x5, p5 = kalman_torch(1, 4, Z, x4, p4, aas[4], zs[4])
    # print({'x5': x5, 'p5': p5})


def kalman_with_random_data():
    """Verify against ground truth [-3, 9, -4, -5]."""

    print("----------------------------------------------------------------")
    print("Recurrence over large-ish data set (10_000 trials).")

    ground_truth = torch.tensor([[-3.0], [9.0], [-4.0], [-5.0]])

    x0 = torch.tensor([[x] for x in torch.zeros(4)])

    p0 = 1000. * torch.eye(4)

    Z = torch.tensor([[1.0]])

    # foldable; lifted over b, n, Z
    fk = lambda xp, az: kalman_torch(1, 4, Z, xp[0], xp[1], az[0], az[1])

    seed = torch.random.initial_seed()
    print({'seed': seed})

    trials = 10_000

    trs = [torch.rand(1) * 4.0 - 2.0 for _ in range(trials)]
    aars = [torch.tensor([[1.0, t, t ** 2, t ** 3]]) for t in trs]
    zrs = [torch.add(torch.matmul(a, ground_truth), torch.randn(1)) for a in aars]

    xtrials, ptrials = toolz.reduce(fk, list(zip(aars, zrs)), [x0, p0])
    print({'xtrials': xtrials, 'ptrials': ptrials})


def kalman_numpy_sample_by_hand():
    """Verify against equation 1 in https://vixra.org/pdf/1606.0328v1.pdf"""

    # print("----------------------------------------------------------------")
    # print("Explicit intermediate variables in a recurrence over five data (numpy version)")

    numpy_type = np.float64
    x0 = np.zeros(4, dtype=numpy_type)

    zs = np.array([[z] for z in [-2.28442, -4.83168, -10.4601, 1.40488, -40.8079]], dtype=numpy_type)

    aas = np.array([[a] for a in [[1., 0., 0., 0.],
                                  [1., 1., 1., 1.],
                                  [1., -1., 1., -1.],
                                  [1., -2., 4., -8.],
                                  [1., 2., 4., 8.]]], dtype=numpy_type)

    p0 = 1000. * np.eye(4, 4, dtype=numpy_type)

    Z = np.array([[1.0]], dtype=numpy_type)

    x1, p1 = kalman_numpy(1, 4, Z, x0, p0, aas[0], zs[0])
    # print({'x1': x1, 'p1': p1})

    x2, p2 = kalman_numpy(1, 4, Z, x1, p1, aas[1], zs[1])
    # print({'x2': x2, 'p2': p2})

    x3, p3 = kalman_numpy(1, 4, Z, x2, p2, aas[2], zs[2])
    # print({'x3': x3, 'p3': p3})

    x4, p4 = kalman_numpy(1, 4, Z, x3, p3, aas[3], zs[3])
    # print({'x4': x4, 'p4': p4})

    x5, p5 = kalman_numpy(1, 4, Z, x4, p4, aas[4], zs[4])
    # print({'x5': x5, 'p5': p5})


def kalman_numpy(
        b,  # # rows, cols, in Z; # rows in z
        n,  # # rows, cols, in P; # rows in x
        Z,  # b x b observation covariance
        x,  # n x 1, current state
        P,  # n x n, current covariance
        A,  # b x n, current observation partials
        z  # b x 1, current observation vector
):
    """Recurrent Kalman filter for parameter estimation (no dynamics)."""

    # Transcribe the following sketch of Wolfram code (the intermediate
    # matrices are not necessary in Wolfram, but we need them in Python).
    #
    # noInverseKalman[Z_][{x_, P_}, {A_, z_}] :=
    #
    #   Module[{PAT, D, Res, DiRes, KRes, AP, DiAP, KAP},
    #
    #    1. PAT    = P.Transpose[A]         (* n x b *)
    #    2. D      = Z + A.PAT              (* b x b *)
    #    3. Res    = z - A.x                (* b x 1 *)
    #    4. DiRes  = LinearSolve[D, Res]    (* b x 1 *)
    #    5. KRes   = PAT.DiRes              (* n x 1 *)
    #    6. AP     = A.P                    (* n x 1 *)
    #    7. DiAP   = LinearSolve[D, AP]     (* b x n *)
    #    8. KAP    = PAT.DiAP               (* n x n *)
    #    9. x     <- x + KRes
    #   10. P     <- P - KAP
    #

    #
    #      PAT                P           AT
    #       b                 n           b
    #    / * * \         / * * * * \   / * * \
    #  n | * * |  <--  n | * * * * | n | * * |
    #    | * * |         | * * * * |   | * * |
    #    \ * * /         \ * * * * /   \ * * /

    pat = np.matmul(P, np.transpose(A))

    #
    #       D                 A          PAT           Z
    #       b                 n           b            b
    #  b / * * \  <--  b / * * * * \ n / * * \  + b / * * \
    #    \ * * /         \ * * * * /   | * * |      \ * * /
    #                                  | * * |
    #                                  \ * * /

    d = np.add(np.matmul(A, pat), Z)

    #            NOTA BENE
    #                |
    #     Res        |        A          x          z
    #      1         v        n          1          1
    #  b / * \  <--  - b / * * * * \ n / * \  + b / * \
    #    \ * /           \ * * * * /   | * |      \ * /
    #                                  | * |
    #                                  \ * /

    res = np.subtract(z, np.matmul(A, x))

    #
    #    DiRes        Di = D^-1   Res
    #      1              b        1
    #  b / * \  <--  b / * * \ b / * \
    #    \ * /         \ * * /   \ * /

    di = np.linalg.inv(d)
    dires = np.matmul(di, res)

    #
    #     KRes           PAT     DiRes
    #      1              b        1
    #  n / * \       n / * * \ b / * \
    #    | * |  <--    | * * |   \ * /
    #    | * |         | * * |
    #    \ * /         \ * * /

    kres = np.matmul(pat, dires)

    #
    #         AP                  A             P
    #         n                   n             n
    #  b / * * * * \  <--  b / * * * * \ n / * * * * \
    #    \ * * * * /         \ * * * * /   | * * * * |
    #                                      | * * * * |
    #                                      \ * * * * /

    ap = np.matmul(A, P)

    #
    #        DiAP           Di = D^-1       AP
    #         n                 b           n
    #  b / * * * * \  <--  b / * * \ b / * * * * \
    #    \ * * * * /         \ * * /   \ * * * * /

    diap = np.matmul(di, ap)

    #
    #        KAP             PAT         DiAP
    #         n               b           n
    #  n / * * * * \  <--  / * * \ b / * * * * \
    #    | * * * * |     n | * * |   \ * * * * /
    #    | * * * * |       | * * |
    #    \ * * * * /       \ * * /

    kap = np.matmul(pat, diap)

    #
    #      x            x          KRes
    #      1            1           1
    #  n / * \  <-- n / * \  + n / * \
    #    | * |        | * |      | * |
    #    | * |        | * |      | * |
    #    \ * /        \ * /      \ * /

    x = np.add(x, kres)

    #                  NOTA BENE
    #                      |
    #         P            |       KAP               P
    #         n            v        n                n
    #  n / * * * * \  <--  - n / * * * * \  + n / * * * * \
    #    | * * * * |           | * * * * |      | * * * * |
    #    | * * * * |           | * * * * |      | * * * * |
    #    \ * * * * /           \ * * * * /      \ * * * * /

    p = np.subtract(P, kap)

    return x, p


if __name__ == "__main__":
    # normal_equations()
    iterations = 10000
    print(f"Torch Kalman, {iterations} iterations.")
    with Timer():
        for _ in range(iterations):
            kalman_torch_sample_by_hand()
    # kalman_with_random_data()
    print(f"Numpy Kalman, {iterations} iterations.")
    with Timer():
        for _ in range(iterations):
            kalman_numpy_sample_by_hand()
