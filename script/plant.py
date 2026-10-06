from plotter import Plotter
from network import PolicyNetwork, QNetwork
import torch
import torch.nn as nn
import numpy as np
import random


class Plant1D:
    def __init__(self, mass, damper, spring, friction, frequency):
        self.mass = mass
        self.damper = damper
        self.spring = spring
        self.friction = friction
        self.frequency = frequency
        self.dt = 1 / frequency
        self.pos = 0.0
        self.vel = 0.0

    def reset(self, pos, vel):
        self.pos = pos
        self.vel = vel

    def step(self, force):
        acc = (force - self.damper * self.vel - self.spring * self.pos - np.tanh(self.vel / 1.0) * self.friction * self.mass * 9.81)/self.mass
        self.vel += acc * self.dt
        self.pos += self.vel * self.dt
        return self.pos, self.vel

class PDController:
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def update(self, kp, kd):
        self.kp = kp
        self.kd = kd

    def setref(self, pos, vel):
        self.pos_ref = pos
        self.vel_ref = vel

    def output(self, pos, vel):
        p_term = self.kp * (self.pos_ref - pos)
        d_term = self.kd * (self.vel_ref - vel)
        return p_term + d_term

class ParaTuningEnv:
    def __init__(self, plant, controller, frequency):
        self.plant = plant
        self.controller = controller
        self.frequency = frequency
        self.period = 1 / frequency
        self.previous_deviation = None

    def get_observation(self):
        pos_error = self.controller.pos_ref - self.plant.pos
        vel_error = self.controller.vel_ref - self.plant.vel

        self.observation = [
            pos_error + np.random.normal(0.0, 0.02),
            vel_error + np.random.normal(0.0, 0.05),
        ]

        return self.observation

    def get_reward(self):
        return self.pos_reward() + 0.1 * self.vel_reward()

    def pos_reward(self):
        error = self.controller.pos_ref - self.plant.pos
        return -error * error

    def vel_reward(self):
        error = self.controller.vel_ref - self.plant.vel
        return -error * error

    def reset(self):
        pass

    def step(self, action):
        
        kp = action[0]
        kd = action[1]

        kp = max(10.0, min(kp, 50.0))
        kd = max(1.0, min(kd, 5.0))

        self.controller.update(kp, kd)
       
class Agent:

    def __init__(self):
        self.policy_network = PolicyNetwork()
        self.q_network = QNetwork()   
        self.target_q_network = QNetwork()
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(),
            lr=0.001
        )
        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=0.001
        )

        self.gamma = 0.99
        self.tau = 0.005

    def policy(self, observation, mode):
        observation = torch.tensor(
            observation,
            dtype=torch.float32
        )
        if mode == 0:
            action = self.policy_network(observation)
  
            return action.detach().numpy()
        
        else:

            action = np.array([
                np.random.uniform(10, 50),
                np.random.uniform(1, 5)
            ])
            return action



    def learn(self, batch, mode):
        observations = torch.tensor(np.array([item[0] for item in batch]), dtype=torch.float32)
        actions = torch.tensor(np.array([item[1] for item in batch]), dtype=torch.float32)
        rewards  = torch.tensor(np.array([item[2] for item in batch]), dtype=torch.float32).unsqueeze(1)
        next_observations = torch.tensor(np.array([item[3] for item in batch]), dtype=torch.float32)
    

        with torch.no_grad():
            next_actions = self.policy_network(next_observations)
            next_q = self.target_q_network(next_observations, next_actions)
            target_q = rewards + self.gamma * next_q

        current_q = self.q_network(observations, actions)
        q_loss = (current_q - target_q).pow(2).mean()

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        with torch.no_grad():
            for target_param, param in zip(
                self.target_q_network.parameters(),
                self.q_network.parameters()
            ):
                target_param.data.copy_(
                    (1.0 - self.tau) * target_param.data
                    + self.tau * param.data
                )

        if mode != 1:
            policy_actions = self.policy_network(observations)
            policy_q = self.q_network(observations, policy_actions)
            policy_loss = -policy_q.mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        

        # print(
        #     "reward:", rewards.mean().item(),
        #     "current_q:", current_q.mean().item(),
        #     "target_q:", target_q.mean().item(),
        #     "q_loss:", q_loss.item()
        #     )


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, observation, action, reward, next_observation):
        self.buffer.append(
            (observation, action, reward, next_observation)
        )

        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

if __name__ == "__main__":
    seed = 0

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


    sim_fre = 100
    learn_fre = 10

    plant = Plant1D(1, 0.05, 0, 0, sim_fre)
    controller = PDController(30, 3)
    env = ParaTuningEnv(plant, controller, learn_fre)
    agent = Agent()
    buffer = ReplayBuffer(10000)

    plotter = Plotter()

    interval = int(sim_fre / learn_fre)
    extForce = 0
    posRef = 10  
    velRef = 0      

    seed = 0

    episode_reward = 0.0
    for episode in range(100):
        posRef = 10
        velRef = 0

        # reset simulation
        plant.reset(10, 10*2*np.pi*0.2)
        controller.update(30, 3)
        controller.setref(10, 0)

        count = 0

        time = []
        pos = []
        vel = []
        pos_ref = []
        vel_ref = []
        ext_force = []
        kp = []
        kd = []

        observation = env.get_observation()

        if episode == 20:
            print("start critic pre-training")

            for i in range(100):
                batch = buffer.sample(64)
                agent.learn(batch, 1)

            print("critic pre-training finished")
            # test_obs = torch.tensor(
            #     [10.0, 0.0, 10.0, 1.0],
            #     dtype=torch.float32
            # )

            # with torch.no_grad():
            #     for delta_kp in np.linspace(-1.0, 1.0, 21):
            #         test_action = torch.tensor(
            #             [delta_kp, 0.0],
            #             dtype=torch.float32
            #         )

            #         q = agent.q_network(test_obs, test_action)

            #         print(
            #             "delta_kp:",
            #             round(delta_kp, 2),
            #             "Q:",
            #             q.item()
            #         )
            # break
            


        if episode < 20:
            action = agent.policy(observation, 1)
        else:
            action = agent.policy(observation, 0)
        env.step(action)
         


            # if count == 600:
               # if count % 10 == 0:
            #     plant.friction = np.random.uniform(0.0, 30.0)
            #     #posRef = np.random.uniform(0.0, 20.0)
            #     plant.friction = 15
        while count < 1000:

            posRef = 10 + 10 * np.sin(2*np.pi*count/500)
            velRef = 10*2*np.pi*0.2*np.cos(2*np.pi*count/500)
            
            # if count % 10 == 0:
            #     plant.friction = np.random.uniform(0.0, 30.0)
            # if plant.pos < 10:
            #     plant.friction = np.random.uniform(0.0, 0.1)
            # else:
            #     plant.friction = np.random.uniform(0.1, 0.2)
            # if count == 300:
          
            #     #posRef = np.random.uniform(0.0, 20.0)
            #     plant.friction = 30

            # if count == 600:
              
            #     #posRef = np.random.uniform(0.0, 20.0)
            #     plant.friction = 15

            controller.setref(posRef, velRef)
            # if count % 50 == 0:
            #     extForce = np.random.uniform(-150.0, 150.0)
                #posRef = 10 + np.random.uniform(-1.0, 1.0)
            plant.step(controller.output(plant.pos, plant.vel) + extForce)
            count += 1

            if count % interval == 0:
                reward = env.get_reward()
                next_observation = env.get_observation()

                

                
                if episode < 20:
                    buffer.add(
                        observation,
                        action,
                        reward,
                        next_observation
                    )
                    # agent.learn(
                    #     observation,
                    #     action,
                    #     reward,
                    #     next_observation,
                    #     0
                        
                    # )
            
                    observation = next_observation
                    action = agent.policy(observation, 1)
                else:
                    buffer.add(
                        observation,
                        action,
                        reward,
                        next_observation
                    )
                    batch = buffer.sample(64)
                    agent.learn(batch, 0)
                    observation = next_observation
                    action = agent.policy(observation, 0)
                
                env.step(action)

            time.append(count * plant.dt)
            pos.append(plant.pos)
            vel.append(plant.vel)
            pos_ref.append(posRef)
            vel_ref.append(velRef)
            ext_force.append(extForce)
            kp.append(controller.kp)
            kd.append(controller.kd)

        print(
            "episode:", episode,
            "pos:", plant.pos,
            "vel:", plant.vel,
            "kp:", controller.kp,
            "kd:", controller.kd
            )
        print(controller.output(plant.pos, plant.vel) + extForce)
      

    # plot the LAST episode
    plotter.plot(0, time, pos, label="Pos", color="red")
    plotter.plot(0, time, pos_ref, label="Pos_Ref", linestyle="--", color="green")
    plotter.plot(0, time, vel, label="Vel", color="blue")
    plotter.plot(0, time, vel_ref, label="Vel_Ref", linestyle="--", color="orange")
    plotter.plot(1, time, ext_force, label="Ext_Force", color="green")
    plotter.plot(2, time, kp, label="Kp", color="Blue")
    plotter.plot(2, time, kd, label="Kd", color="green")
    plotter.show()