import pandas as pd
import xarray as xr
import scipy as sp
import matplotlib.pyplot as plt
import brian2 as br2
import brian2.numpy_ as np
import names


class VisualStimData:


    """A class holding the experimental dataBase including analysis methods"""
    "This class stores data recieved to it via the method add_instance"


    def __init__(self ):
            self.data = xr.Dataset()

    def add_instance(self, rat_id='' , temp='', humidity='', experimenter_name='',
                        rat_gender='', measured_voltage='', stimuli_index=''):
        """ input -
            rat_id : int, temp: array(4, 10000), humidity:array(4, 10000), experimenter_name: str,
             rat_gender: str, measured_voltage: array(10,10000,4), stimuli_index: array(10000)"""
        try:
            da_instance = xr.DataArray(measured_voltage, name=str(rat_id), dims=['electrode', 'sample', 'repetition'],
                        coords={
                               'temperature': (['repetition', 'sample'], temp),
                               'humidity': (['repetition', 'sample'], humidity),
                               'stimuli_index': (['sample'], stimuli_index)},
                        attrs={'rat_gender': rat_gender,
                               'experimenter_name': experimenter_name,
                                'rat_id': rat_id})
            # pre calculating descriptors
            da_instance.attrs['mean'] = da_instance.mean().values
            da_instance.attrs['median'] = da_instance.median().values
            da_instance.attrs['std'] = da_instance.std().values

            self.data[rat_id] = da_instance
        except(TypeError, ValueError):
            'Can not add\create mock data instance'

    def plot_electrode(self, rep_number=1, rat_id=0, elec_number=(0, 1)):
        """
        Plots the voltage of the electrodes in "elec_number" for the rat "rat_id" in the repetition
        "rep_number". Shows a single figure with subplots.
        """
        n_of_elec = len(elec_number)
        # creating time vector
        time_vec = np.linspace(0, 2000, 10000)
        # subplot instances
        fig, all_ax = plt.subplots(n_of_elec, 1)
        # removing x ticks data for all electrodes except the last one
        [plt.setp( all_ax[i].get_xticklabels(), visible=False) for i in range(0, n_of_elec-1)]
        # plotting the data
        [all_ax[i].plot(time_vec, self.data[rat_id].sel(repetition=rep_number, electrode=elec).values)
         for elec, i in zip(elec_number, range(0, n_of_elec))]
        # setting labels
        all_ax[-1].set_xlabel('Time [ms]')
        [all_ax[i].set_ylabel('Amplitude') for i in range(0, n_of_elec)]
        # setting titles
        [all_ax[i].set_title('electrode # ' + str(elec)) for i, elec in zip(range(0, n_of_elec), elec_number)]
        plt.show()

    def experimenter_bias(self):
        """ Shows the statistics of the average recording across all experimenters """
        # extracting rat id list
        id_list = [rat_id for rat_id in self.data.data_vars]
        # extracting experimenters name list - conversion for set and back to list for uniqueness
        exp_name_list = [self.data[id].experimenter_name for id in id_list]
        # extracting mean, average and std
        descriptors = [(self.data[id].mean(), self.data[id].std(), self.data[id].median()) for id in id_list]
        # creating dataFrame variables
        avrg = np.array(descriptors)[:, 0]
        std = np.array(descriptors)[:, 1]
        med = np.array(descriptors)[:, 2]
        # dataframe instance
        plt.figure()
        df = pd.DataFrame({'average': avrg, 'std': std, 'median': med}, index=exp_name_list)
        # barplot
        df.plot.bar(rot=0)
        plt.show()
    pass

def mock_stim_data():
    """ Creates a new VisualStimData instance with mock data
    adds data to a VisualStimData object using its add_instance method
    output: VisualStimData object"""
    # constructing an instance of VisualStimData
    vsd = VisualStimData()
    # adding 3 instances of mock data
    [vsd.add_instance(*create_instance(id)) for id in range(0, 3)]
    return vsd


def create_instance(rat_id=0):
    """ Creates a mock data instance
    input-
     rat_id: int
     output -
     tuple containing a single mock data instance"""

    # simulating the neuron
    measured_voltage = neuron_sim()
    # generating random temperature
    temp = np.random.rand(4, 10000)+25
    # generating random humidity
    humidity = np.random.rand(4, 10000)+50
    # generating random experimenter name
    experimenter_name = names.get_full_name()
    # generating random rat gender
    rat_gender = randomized_gender()
    # preparing stimuli index
    stimuli_index = np.ones(shape=10000)
    stimuli_index[0:4999] = 0
    stimuli_index[5000:5500] = 1
    stimuli_index[5501:9999] = 2

    return rat_id, temp, humidity, experimenter_name, rat_gender, measured_voltage, stimuli_index


def neuron_sim(stim_V=0.8):
    """simulating the neuron using brian2 library"""
    # initiate stimuli timing
    stimulus = br2.TimedArray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, stim_V, 0, 0, 0, 0, 0, 0, 0, 0, 0], dt=100*br2.ms)
    # initiating neurons
    eqs = '''
    dv/dt =  (-v + stimulus(t)-0.1*rand()) / tau  : 1
    tau : second
    '''
    # creating neuron group
    G = br2.NeuronGroup(10, eqs, dt=0.2*br2.ms)
    # creating different tau's for the neuron model
    G.tau = np.linspace(10, 100, 10)*br2.ms
    # defining state monitor to record voltage
    M = br2.StateMonitor(G, 'v', record=True)
    # creating network
    net = br2.Network(G, M)
    # storing initial state
    net.store()
    # np array to contain data
    data = np.zeros(shape=(10, 10000, 4))
    # producing four repetitions
    for trial in range(4):
        net.restore()  # Restore the initial state
        net.run(2*br2.second)
        # store the results
        data[:, :, trial] = M.v

    return data



def randomized_gender():
    genders = ['Male', 'Female']
    return genders[np.random.randint(low=0, high=2)]


if __name__ == '__main__':
    vsd = mock_stim_data()
    vsd.experimenter_bias()
    vsd.plot_electrode(rep_number=3, rat_id=1, elec_number=(0, 1, 2, 3, 6))
