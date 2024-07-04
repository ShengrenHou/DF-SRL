

import numpy as np
import pandas as pd
import gym
from gym import spaces
import pandas as pd
import pandapower as pp
import copy as cp
from scipy.sparse import csr_matrix
from network import create_cigre_lv_resident
import pyomo.environ as pyo
from pyomo.opt import SolverFactory


battery_parameters={
'capacity':3000,# kW.h
'max_charge':5.0, # kW
'max_discharge':5.0, #kW
'efficiency':1,
'degradation':0, #euro/kw
'max_soc':0.8,
'min_soc':0.2,
'initial_soc':0.4}
class Battery():
    '''simulate a simple battery here'''
    def __init__(self,parameters):
        self.capacity=parameters['capacity']#
        self.max_soc=parameters['max_soc']# max soc 0.8
        self.initial_soc=parameters['initial_soc']# initial soc 0.4
        self.min_soc=parameters['min_soc']# 0.2
        self.degradation=parameters['degradation']# degradation cost 0，
        self.max_charge=parameters['max_charge']# max charge ability
        self.max_discharge=parameters['max_discharge']# max discharge ability
        self.efficiency=parameters['efficiency']# charge and discharge efficiency

    def step(self,action_battery):

        energy=action_battery*self.max_charge

        updated_soc=max(self.min_soc,min(self.max_soc,(self.current_soc*self.capacity+energy*5/60)/self.capacity))

        self.energy_change=(updated_soc-self.current_soc)*self.capacity*12# if charge, positive, if discharge, negative
        # print(self.energy_change)
        self.current_soc=updated_soc# update capacity to current codition
    def _get_cost(self,energy):# calculate the cost depends on the energy change
        # cost=energy**2*self.degradation
        cost=np.abs(energy)
        return cost
    def SOC(self):
        return self.current_soc
    def reset(self):
        # self.current_soc=self.initial_capacity
        self.current_soc=0.4
        # self.current_soc=0.2
        # self.current_soc=np.random.uniform(0.2,0.8)
class Constant:
    MONTHS_LEN = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    MAX_STEP_HOURS = 24 * 30
class DataManager():
    def __init__(self) -> None:
        self.PV_Generation = []
        self.Prices = []
        self.Electricity_Consumption = []

    def add_pv_element(self, element): self.PV_Generation.append(element)

    def add_price_element(self, element): self.Prices.append(element)

    def add_electricity_element(self, element): self.Electricity_Consumption.append(element)

    # maybe we also need to add year variable, get current time data based on given month day, and day_time
    def get_pv_data(self, month, day, day_time): return self.PV_Generation[
        (sum(Constant.MONTHS_LEN[:month - 1]) + day - 1) * 96 + day_time]

    def get_price_data(self, month, day, day_time): return self.Prices[
        (sum(Constant.MONTHS_LEN[:month - 1]) + day - 1) * 96 + day_time]

    def get_electricity_cons_data(self, month, day, day_time): return self.Electricity_Consumption[
        (sum(Constant.MONTHS_LEN[:month - 1]) + day - 1) * 96 + day_time]

    # get series data for one episode
    def get_series_pv_data(self, month, day): return self.PV_Generation[
                                                     (sum(Constant.MONTHS_LEN[:month - 1]) + day - 1) * 24:(
                                                                                                                   sum(Constant.MONTHS_LEN[
                                                                                                                       :month - 1]) + day - 1) * 24 + 24]

    def get_series_price_data(self, month, day): return self.Prices[
                                                        (sum(Constant.MONTHS_LEN[:month - 1]) + day - 1) * 24:(
                                                                                                                      sum(Constant.MONTHS_LEN[
                                                                                                                          :month - 1]) + day - 1) * 24 + 24]

    def get_series_electricity_cons_data(self, month, day): return self.Electricity_Consumption[(
                                                                                                        sum(Constant.MONTHS_LEN[
                                                                                                            :month - 1]) + day - 1) * 24:(
                                                                                                                                                 sum(Constant.MONTHS_LEN[
                                                                                                                                                     :month - 1]) + day - 1) * 24 + 24]
class PowerNetEnv(gym.Env):
    def __init__(self,**kwargs):
        super(PowerNetEnv,self).__init__()
        self.episode_length=kwargs.get('episode_length',96)

        self.net=kwargs.get('power_net',create_cigre_lv_resident())
        self.action_space=spaces.Box(low=-1,high=1,shape=(18,),dtype=np.float32)
        self.state_space=spaces.Box(low=0,high=1,shape=(36,),dtype=np.float32)
        self.Length_max=96
        self.current_time=None
        self.year=None
        self.month=None
        self.day=None
        self.data_manager=DataManager()
        self._load_data()
        self.Train=True
        ## set all batteries
        dic={'battery_1':Battery(battery_parameters),'battery_2':Battery(battery_parameters),'battery_3':Battery(battery_parameters),'battery_4':Battery(battery_parameters),'battery_5':Battery(battery_parameters),'battery_6':Battery(battery_parameters),
             'battery_7':Battery(battery_parameters),'battery_8':Battery(battery_parameters),'battery_9':Battery(battery_parameters),'battery_10':Battery(battery_parameters),'battery_11':Battery(battery_parameters),'battery_12':Battery(battery_parameters),
             'battery_13':Battery(battery_parameters),'battery_14':Battery(battery_parameters),'battery_15':Battery(battery_parameters),'battery_16':Battery(battery_parameters),'battery_17':Battery(battery_parameters),'battery_18':Battery(battery_parameters),}
        for i in dic.keys():
            setattr(self,i,dic[i])
    def init(self):
        self.p_max=self.battery_1.max_charge/1000
        self.v_max=1.05
        self.v_min=0.95
        _,_,_,_,_=self._get_net_info()
    def get_safe_action(self,action):
        p = np.array(self.net.load['p_mw']).reshape(-1, 1)
        vector_1=np.ones((p.shape[0], 1)).reshape(-1, 1)
        M=self.M
        B=self.B
        D_r=self.D_r
        T=self.T
        action=action.reshape(-1,1)
        real_p=p
        square_v = 1.0 * vector_1 + np.linalg.inv(M) @ (2*D_r @ B @ T[:, 1:] @ real_p)
        linear_v=np.sqrt(square_v).flatten()
        real_v = np.array(self.net.res_bus.vm_pu)[2:]
        error = real_v - linear_v


        inv_M=np.linalg.inv(M)
        S=2*D_r @ B @ T[:, 1:]
        A=inv_M@S

        m=pyo.ConcreteModel()
        N,L=M.shape[0],M.shape[1]
        m.N = pyo.Set(initialize=(range(N)), ordered=False)
        m.L = pyo.Set(initialize=(range(L)), ordered=False)
        m.A=pyo.Param(m.N,m.L,mutable=True, within=pyo.Reals)
        for i in m.N:
            for j in m.L:
                m.A[i,j]=A[i,j]

        m.p=pyo.Param(m.N,initialize=p,mutable=False)
        m.p_max=pyo.Param(default=self.p_max,mutable=False)
        m.action=pyo.Var(m.N,initialize=action,bounds=(-1,1))
        def lower_boundary_rule(m,i):
            '''implement for 1.0 * vector_1+A@real_p'''
            return 1.0 +sum(m.A[i,j]*(m.p[j]+m.action[j]*m.p_max) for j in m.L)>=(0.952**2)

        m.lb_cons = pyo.Constraint(m.N, rule=lower_boundary_rule)
        def upper_boundary_rule(m, i):
            '''implement for 1.0 * vector_1+A@real_p '''
            return 1.0 +sum(m.A[i,j]*(m.p[j]+m.action[j]*m.p_max) for j in m.L)<=(1.048 ** 2)

        m.ub_cons = pyo.Constraint(m.N, rule=upper_boundary_rule)

        def obj_rule(m):
            return sum((m.action[i]-action[i])**2 for i in m.N)
        # m.pprint()

        m.obj = pyo.Objective(expr=obj_rule, sense=pyo.minimize)

        pyo.SolverFactory('gurobi').solve(m, tee=False)
        '''here we need to check the printed model for each constrain, are they satisfied for the real part'''

        safe_action=pyo.value(m.action[:])
        return safe_action
    def _get_net_info(self):
        '''We get the information of this network including A, LB,UB,P,Q'''
        self.reset()

        self._build_state()
        data=self.net._ppc
        bus = data['bus']
        branch = data['branch']
        nb = bus.shape[0] - 1
        nl = branch.shape[0]
        f = np.real(branch[:, 0]).astype(int)  ## list of "from" buses
        t = np.real(branch[:, 1]).astype(int)  ## list of "to" buses
        Cf = csr_matrix((np.ones(nl), (range(nl), f)), (nl, nb))
        Ct = csr_matrix((np.ones(nl), (range(nl), t)), (nl, nb))
        T = Ct.toarray()
        F = Cf.toarray()
        M_0 = F - T
        m_0 = M_0[:, 0]
        M = M_0[:, 1:]
        I = np.identity(M.shape[0])
        B = np.linalg.inv(I - T @ np.transpose(F))
        stat = branch[:, 10]  ## ones at in-service branches
        D_r = np.real(np.diag(branch[:, 2]))
        D_x = np.real(np.diag(branch[:, 3]))
        p = np.array(self.net.load['p_mw']).reshape(-1, 1)
        q = np.array(self.net.load['q_mvar']).reshape(-1, 1)


        vector_1 = np.ones((M.shape[0], 1)).reshape(-1, 1)
        square_v = 1.0 * vector_1 + 2 * np.linalg.inv(M) @ (D_r @ B @ T[:, 1:] @ p + D_x @ B @ T[:, 1:] @ q)
        linear_v = np.sqrt(square_v).flatten()
        real_v = np.array(self.net.res_bus.vm_pu)[2:]
        error = real_v - linear_v
        ## get the important matrix we formulated

        A = 2 * np.linalg.inv(M) @ D_r @ B @ T[:, 1:]
        # we now count q as parameter (constant)

        LB = ((0.95 ** 2 - 1.0) * vector_1 - 2 * np.linalg.inv(M) @ D_x @ B @ T[:, 1:] @ q)
        UB = ((1.05 ** 2 - 1.0) * vector_1 - 2 * np.linalg.inv(M) @ D_x @ B @ T[:, 1:] @ q)
        ## return the self.information
        self.M=M
        self.D_r=D_r
        self.B=B
        self.T=T
        return A, LB, UB,p,q
    def reset(self):
        '''we need to change 1. current-time 2.identify the first data to the net, 2.the network to intialize condition based on the first data'''
        # we also need a function to build state
        self.year=1
        self.month=np.random.randint(2,13)
        if self.Train:
            self.day = np.random.randint(1, 21)
        else:
            self.day = np.random.randint(21, Constant.MONTHS_LEN[self.month - 1])
        self.current_time=0
        ## change the current battery
        self.battery_1.reset()
        self.battery_2.reset()
        self.battery_3.reset()
        self.battery_4.reset()
        self.battery_5.reset()
        self.battery_6.reset()
        self.battery_7.reset()
        self.battery_8.reset()
        self.battery_9.reset()
        self.battery_10.reset()
        self.battery_11.reset()
        self.battery_12.reset()
        self.battery_13.reset()
        self.battery_14.reset()
        self.battery_15.reset()
        self.battery_16.reset()
        self.battery_17.reset()
        self.battery_18.reset()
        return self._build_state()
    def _build_state(self):
        '''normalize the original information into state and then transfer it into normalized state'''
        for bus_index in self.net.load.bus.index:
            self.net.load.p_mw[bus_index] = self.data_manager.get_electricity_cons_data(self.month,self.day,self.current_time)[bus_index]
            # self.net.load.q_mvar[bus_index] = self.net.load.p_mw[bus_index]*self.randfloat(0.05,0.1)
            self.net.load.q_mvar[bus_index] = self.net.load.p_mw[bus_index] * 0.0
        pp.runpp(self.net,algorithm='nr')
        vm_pu=cp.deepcopy(self.net.res_bus.vm_pu)

        # we only need node1 to node 18
        vm_pu=vm_pu.iloc[2:]
        p_mw_boundary = cp.deepcopy(self.net.load.p_mw).to_numpy(dtype=float)
        soc_1=self.battery_1.SOC()
        soc_2=self.battery_2.SOC()
        soc_3=self.battery_3.SOC()
        soc_4=self.battery_4.SOC()
        soc_5=self.battery_5.SOC()
        soc_6=self.battery_6.SOC()
        soc_7=self.battery_7.SOC()
        soc_8=self.battery_8.SOC()
        soc_9=self.battery_9.SOC()
        soc_10=self.battery_10.SOC()
        soc_11=self.battery_11.SOC()
        soc_12=self.battery_12.SOC()
        soc_13=self.battery_13.SOC()
        soc_14=self.battery_14.SOC()
        soc_15=self.battery_15.SOC()
        soc_16=self.battery_16.SOC()
        soc_17=self.battery_17.SOC()
        soc_18=self.battery_18.SOC()
        soc_array=np.array([soc_1,soc_2,soc_3,soc_4,soc_5,soc_6,soc_7,soc_8,soc_9,soc_10,soc_11,soc_12,soc_13,soc_14,soc_15,soc_16,soc_17,soc_18])

        state=np.concatenate((p_mw_boundary,vm_pu.to_numpy(dtype=float)),axis=None)
        self.state=state
        return state
    def step(self,action):
        '''core function for the environment. first take the action, transfer action to real value.
        put value into the net, the net use a function to execute power flow and return the essential information
        based on these information, we check the reward
        '''

        current_obs=self.state
        vm_pu_before_control=current_obs[18:]
        # maybe here can be optimized to reduce computation time,change for circle
        self.battery_1.step(action[0])
        self.battery_2.step(action[1])
        self.battery_3.step(action[2])
        self.battery_4.step(action[3])
        self.battery_5.step(action[4])
        self.battery_6.step(action[5])
        self.battery_7.step(action[6])
        self.battery_8.step(action[7])
        self.battery_9.step(action[8])
        self.battery_10.step(action[9])
        self.battery_11.step(action[10])
        self.battery_12.step(action[11])
        self.battery_13.step(action[12])
        self.battery_14.step(action[13])
        self.battery_15.step(action[14])
        self.battery_16.step(action[15])
        self.battery_17.step(action[16])
        self.battery_18.step(action[17])
        ## charge is positive and discharge is negative
        energy_0=self.battery_1.energy_change
        energy_1=self.battery_2.energy_change
        energy_2=self.battery_3.energy_change
        energy_3=self.battery_4.energy_change
        energy_4=self.battery_5.energy_change
        energy_5=self.battery_6.energy_change
        energy_6=self.battery_7.energy_change
        energy_7=self.battery_8.energy_change
        energy_8=self.battery_9.energy_change
        energy_9=self.battery_10.energy_change
        energy_10=self.battery_11.energy_change
        energy_11=self.battery_12.energy_change
        energy_12=self.battery_13.energy_change
        energy_13=self.battery_14.energy_change
        energy_14=self.battery_15.energy_change
        energy_15=self.battery_16.energy_change
        energy_16=self.battery_17.energy_change
        energy_17=self.battery_18.energy_change
        ## put these part of energy to the grid, maybe check here the powe
        self.net.load.p_mw[0]+=energy_0/1000
        self.net.load.p_mw[1]+=energy_1/1000
        self.net.load.p_mw[2]+=energy_2/1000
        self.net.load.p_mw[3]+=energy_3/1000
        self.net.load.p_mw[4]+=energy_4/1000
        self.net.load.p_mw[5]+=energy_5/1000
        self.net.load.p_mw[6]+=energy_6/1000
        self.net.load.p_mw[7]+=energy_7/1000
        self.net.load.p_mw[8]+=energy_8/1000
        self.net.load.p_mw[9]+=energy_9/1000
        self.net.load.p_mw[10]+=energy_10/1000
        self.net.load.p_mw[11]+=energy_11/1000
        self.net.load.p_mw[12]+=energy_12/1000
        self.net.load.p_mw[13]+=energy_13/1000
        self.net.load.p_mw[14]+=energy_14/1000
        self.net.load.p_mw[15]+=energy_15/1000
        self.net.load.p_mw[16]+=energy_16/1000
        self.net.load.p_mw[17]+= energy_17/1000



        pp.runpp(self.net, algorithm='nr')

        vm_pu_after_control=cp.deepcopy(self.net.res_bus.vm_pu).to_numpy(dtype=float)
        vm_pu_after_control=vm_pu_after_control[2:]
        self.after_control=vm_pu_after_control
        energy_used=np.array([energy_0,energy_1,energy_2,energy_3,energy_4,energy_5,energy_6,energy_7,energy_8,energy_9,energy_10,
                              energy_11,energy_12,energy_13,energy_14,energy_15,energy_16,energy_17])
        p_used_for_regulation=sum(np.absolute(energy_used))
        ## minimize the power used for regulation and penalize the voltage violation
        reward_for_power=-0.001*p_used_for_regulation
        reward_for_penalty=0.0
        reward_for_good_action=0.0
        #implement pedro's paper penalty
        for i in range(18):
            reward_for_penalty+=min(0,1000*(0.05-abs(1.0-vm_pu_after_control[i])))
        # give some positive reward to this silly agent
        for j in range (18):
            # value of this part is too small
            value_before=min(0,0.05-abs(1.0-vm_pu_before_control[j]))
            value_after=min(0,0.05-abs(1.0-vm_pu_after_control[j]))
            reward_for_good_action+=200*(value_after-value_before)

        self.reward_for_power=reward_for_power
        self.reward_for_good_action=reward_for_good_action
        self.reward_for_penalty=reward_for_penalty
        reward=reward_for_power+reward_for_penalty

        finish=(self.current_time==self.episode_length-1)
        self.current_time += 1
        if finish:
            self.current_time=0
            next_obs=self.reset()
        else:
            next_obs=self._build_state()
        return current_obs,next_obs,float(reward),finish
    def render(self, current_obs, next_obs, reward, finish):
        # print('day={}'.format(self.day))
        # print('day={},hour={:2d}, state={}, next_state={}, reward={:.4f}, terminal={}\n'.format(self.day,self.current_time, current_obs, next_obs, reward, finish))
        print('state={}, next_state={}, reward={:.4f}, terminal={}\n'.format(current_obs, next_obs, reward, finish))
        # pass
    def _load_data(self):
        dataload_series=pd.read_csv(("aihui_material/load_data.csv"), header=0)
        data=dataload_series.to_numpy(dtype=float)
        for element in data:
            self.data_manager.add_electricity_element(element)
    def randfloat(self,l, h):
        '''generate random sample with limited boundaries'''
        a = h - l
        b = h - a
        return np.random.rand(1) * a + b
if __name__ == '__main__':
    # test function
    list_bad_dates=[]
    list_next_state=[]
    power_net_env=PowerNetEnv()

    power_net_env.init()



