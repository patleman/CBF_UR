import numpy as np
import proxsuite


class CBFQPSolver:
    def __init__(self, n, n_eq, n_ieq):
        self.n = n
        self.n_eq = n_eq
        self.n_ieq = n_ieq
        self.qp = proxsuite.proxqp.dense.QP(self.n, self.n_eq, self.n_ieq)
        self.initialized = False

    def solve(self, params):
        self.H, self.g, self.C, self.lb = self.compute_params(params)

        if not self.initialized:
            self.qp.init(H=self.H, g=self.g, C=self.C, l=self.lb)
            self.qp.settings.eps_abs = 1.0e-6
            self.initialized = True
        else:
            self.qp.update(H=self.H, g=self.g, C=self.C, l=self.lb)

        self.qp.solve()

    def compute_params(self, params):
        H =2* np.eye(6)

        g = (
            -2 * (params["u_current"]).reshape(-1, 1)
        )

        C0 = (2 * (params["p_error"].T) @ params["Jacobian"]).reshape(1, -1)

        C1 = (2 * (params["p_error1"].T) @ params["Jacobian1"]).reshape(1, -1)

        C2 = (2 * (params["p_error2"].T) @ params["Jacobian2"]).reshape(1, -1)

        C=np.array([C0,C1,C2]).reshape(3,-1)

        
        gamma = 0.3

        lb0 = -gamma * (np.dot(params["p_error"], params["p_error"])).reshape(1, 1)

        lb1 = -gamma * (np.dot(params["p_error1"], params["p_error1"])).reshape(1, 1)

        lb2 = -gamma * (np.dot(params["p_error2"], params["p_error2"])).reshape(1, 1)

        lb=np.array([lb0 , lb1,lb2]).reshape(3,-1)
        lb[0]=lb[0]+0.03
        lb[1]=lb[1]+0.03
        lb[2]=lb[2]+0.03
        # print(lb0,lb1,lb2)


        return H, g, C, lb
    

    