import sys
import pylab as plb
import numpy as np
import mountaincar

class DummyAgent():
    """A not so good agent for the mountain-car task.
    """

    def __init__(self, mountain_car = None, parameter1 = 3.0):
        
        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car

        self.parameter1 = parameter1

    def visualize_trial(self, n_steps = 200):
        """Do a trial without learning, with display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """
        
        # prepare for the visualization
        plb.ion()
        mv = mountaincar.MountainCarViewer(self.mountain_car)
        mv.create_figure(n_steps, n_steps)
        plb.draw()
            
        # make sure the mountain-car is reset
        self.mountain_car.reset()

        for n in range(n_steps):
			
            #print ('\rt =', self.mountain_car.t,
            sys.stdout.flush()
            
            # choose a random action
            #self.mountain_car.apply_force(np.random.randint(3) - 1)
            
            # Let Neural network chose action
            action = self.mountain_car.chose_action()
            self.mountain_car.apply_force(action-1)
            
            
            # simulate the timestep
            self.mountain_car.simulate_timesteps(100, 0.01)

            # update the visualization
            mv.update_figure()
            plb.draw()            
            
            # learn
            #self.mountain_car.learn(action)
            
            # check for rewards
            if self.mountain_car.R > 0.0:
                print("Reward obtained at t = ", self.mountain_car.t)
                self.mountain_car.reset()

    def learn(self, trials = 1):
		
        # make sure the mountain-car is reset
        self.mountain_car.reset()
        t = 0
        duration = np.array(range(trials))
        while t < trials:
            
            # choose a random action
            #self.mountain_car.apply_force(np.random.randint(3) - 1)
            
            # Let Neural network chose action
            action = self.mountain_car.chose_action()
            self.mountain_car.apply_force(action-1)
            
            # simulate the timestep
            self.mountain_car.simulate_timesteps(100, 0.01)
            
            # learn
            self.mountain_car.learn(action)
                
            # check for rewards
            if self.mountain_car.R > 0.0:
                #self.mountain_car.network.save(t)
                print 'Reward ' + str(t+1) + ' obtained at t = ' + str(self.mountain_car.t)
                duration[t]=self.mountain_car.t
                self.mountain_car.reset()
                t += 1
                self.mountain_car.network.eta *= 1.0-0.5/trials
                #self.mountain_car.network.tau *= 0.9
             
        ##Save Network weights to text file
		#self.mountain_car.network.save()
        
        np.savetxt('Memory/durations.txt', duration, fmt='%d')  
        print 'Mean duration is ' + str(np.mean(duration))
        return duration

	
if __name__ == "__main__":
    
    ##Single network trial
    
    d = DummyAgent()
    
    ###Load a saved network from file
    d.mountain_car.network.load()
    
    ###Update the quiver
    ##d.mountain_car.updatequiver = True
    
    ### train network for n trials
    #d.learn(50)
    
    ###vizualize simumlation for time n
    d.visualize_trial(6000)
    
    plb.show()
    



    ##Average duration training protocol for several networks####################################################################
    #iteratios = 50
    #durations = np.array([np.array([0.0 for i in range(iteratios)]) for j in range(10)])
    #for i in range(10):
        #d = DummyAgent()
        #durations[i] = d.learn(iteratios)
        
    #avduratio = np.array(range(iteratios))
    #for i in range(iteratios):
        #avduratio[i] = 0.0
        #for j in range(10):
            #avduratio[i] += durations[j][i]
        
        #avduratio[i] = avduratio[i]/10
        
    #np.savetxt('Memory/avduration.txt', avduratio, fmt='%d')  

