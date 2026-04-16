import numpy as np
import torch
from math import cos, sin, tanh

def get_num_legjoints(robot):
    if robot == 'default':
        robot = 'Slalom'
    try:
        if robot == 'Slalom':
            num_legs = 4
            num_joints = 4
            motor_mapping = torch.tensor([0,  4,  8,  12, 
                                          1,  5,  9,  13,  
                                          2,  6,  10, 14, 
                                          3,  7,  11, 15]
                                        )
    except:
        print("error get_num_legjoints function, please recheck the name of the robot")
    return num_legs, num_joints, motor_mapping

class RBFNet:
    def __init__(self, 
                 popsize, 
                 num_basis,
                 num_output,
                 robot,
                 motor_encode='direct'):
        """
        sizes: [input_size, hid_1, ..., output_size]
        """
        self.architecture = [num_basis, num_output]
        self.popsize = popsize

        # Initialize CPG
        self.O = torch.Tensor([[0.0, 0.18]]).expand(popsize, 2).cuda()
        self.t, self.x, self.y, self.period = self.pre_compute_cpg()

        # RBF networks parameters
        self.num_basis = num_basis
        self.num_output = num_output
        self.variance = 0.04
        self.phase = 0

        # Pre calculated rbf layers output 
        self.ci, self.cx, self.cy, self.rx, self.ry, self.KENNE = self.pre_rbf_centers(
            self.period, self.num_basis, self.x, self.y, self.variance)
        self.KENNE = self.KENNE.cuda()

        # Get number of legs, joints, and motor mapping from model --> sim robot
        self.num_legs, self.num_joints, motor_mapping  = get_num_legjoints(robot)
        self.indices = motor_mapping.cuda()

        # initilize motor encoding type (weights, CPGs' phase)
        self.motor_encode = motor_encode
        if self.motor_encode == 'direct':
            self.weights = torch.Tensor(popsize, num_basis, num_output).uniform_(-0.1, 0.1).cuda()
            # Initilize phase of each CPG
            self.phase = torch.zeros(2).cuda()


    def forward(self, pre):        
        with torch.no_grad():

            # Direct encoding ##################################
            if self.motor_encode == 'direct':
                p = self.KENNE[int(self.phase[0])]
                out = torch.tanh(torch.matmul(p, self.weights))
                post = torch.index_select(out, 1, self.indices)

                self.phase = self.phase + 1
                self.phase = torch.where(self.phase > self.period, 0, self.phase)
            ####################################################

        return post.float().detach()
    

    def get_n_params_a_model(self):
        return len(self.get_a_model_params())

    def get_models_params(self):
        p = torch.cat([ params.flatten() for params in self.weights] )

        return p.cpu().flatten().numpy()

    def get_a_model_params(self):
        p = torch.cat([ params.flatten() for params in self.weights[0]] )

        return p.cpu().flatten().numpy()
    
    def set_models_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params).float()
        # print('flat_params: ', flat_params.shape)

        popsize, basis, num_out = self.weights.shape
        self.weights = flat_params.reshape(popsize, basis, num_out).cuda()

    def set_a_model_params(self, flat_params):
        flat_params = torch.from_numpy(flat_params).float()
        # print('flat_params: ', flat_params.shape)

        popsize, basis, num_out = self.weights.shape
        # print('flat_params.repeat(popsize, 1, 1): ', flat_params.repeat(popsize, 1, 1).shape)
        self.weights = flat_params.repeat(popsize, 1, 1).reshape(popsize, basis, num_out).cuda()
            
    
    def pre_compute_cpg(self):
        # Run for one period
        phi   = 0.06*np.pi      # SO(2) Frequency
        alpha = 1.01            # SO(2) Alpha term
        w11   = alpha*cos(phi)
        w12   = alpha*sin(phi)
        w21   =-w12
        w22   = w11
        x     = []
        y     = []
        t     = []
        t.append(0)
        x.append(-0.197)
        y.append(0.0)
        period = 0
        while y[period] >= y[0]:
            period = period+1
            t.append(period*0.0167)
            x.append(tanh(w11*x[period-1]+w12*y[period-1]))
            y.append(tanh(w22*y[period-1]+w21*x[period-1]))
            
        while y[period] <= y[0]:
            period = period+1
            t.append(period*0.0167)
            x.append(tanh(w11*x[period-1]+w12*y[period-1]))
            y.append(tanh(w22*y[period-1]+w21*x[period-1]))
        period = period
        return t, x, y, period
    
    def pre_rbf_centers(self, period, num_basis, x, y, var):
        KENNE  = [0]*num_basis  # Kernels
        ci = np.asarray(np.around(np.linspace(1, period, num_basis+1)), dtype=int)

        ci = ci[:-1]

        cx = [0] * (len(ci))
        cy = [0] * (len(ci))

        for k in range(len(ci)):
            cx[k] = x[ci[k]]
            cy[k] = y[ci[k]]

        for i in range(num_basis):
            rx   = [q - cx[i] for q in x]
            ry   = [q - cy[i] for q in y]
            KENNE[i] = np.exp(-(np.power((rx),2) + np.power((ry),2))/var)

        return ci, cx, cy, rx, ry, torch.from_numpy(np.array(KENNE).T).float()