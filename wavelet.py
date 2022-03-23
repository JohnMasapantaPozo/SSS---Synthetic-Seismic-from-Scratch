import warnings
import numpy as np
import bruges
import matplotlib.pyplot as plt

# Wavelet contructor function

def construct_wavelet(duration, sampling_rate, freq, wavelet_type='ricker'):
    """ Wavelet contructor.

    Args:
        duration (flloat): Wavelet duration in seconds.
        samplin_rate (float): Sampling interval in seconds, normally 0.001. 
        freq (int): Central wavelet frequency in Hz.
        wavelet_type: Can be 'ricker', 'gabor', 'klauder' or 'ormsby'
        
    Returns:
        If return_t set to True. It returns a tuple of wavelet sampled times.
    """
    wave_freq = {'ormsby_freq': [10, 20, 20, 40], 
                 'klauder_freq': [25, 40]}
    
    if wavelet_type == 'ricker':
        w_amplitude, w_time = bruges.filters.wavelets.ricker(duration,
                                            dt=sampling_rate, 
                                            f=freq,
                                            t=None,
                                            return_t=True,
                                            sym=True)
    
    elif wavelet_type == 'gabor':
        w_amplitude, w_time = bruges.filters.wavelets.gabor(duration,
                                            dt = sampling_rate,
                                            f=freq,
                                            t=None,
                                            return_t=True,
                                            sym=True)
        
    elif wavelet_type == 'klauder':
        freq = wave_freq['klauder_freq']
        w_amplitude, w_time = bruges.filters.wavelets.klauder(duration,
                                            dt = sampling_rate,
                                            f=freq,
                                            t=None,
                                            return_t=True,
                                            sym=True)
        
    elif wavelet_type == 'ormsby':
        freq = wave_freq['ormsby_freq']
        w_amplitude, w_time = bruges.filters.wavelets.ormsby(duration,
                                        dt = sampling_rate,
                                        f=freq,
                                        t=None,
                                        return_t=True,
                                        sym=True)
    
    return w_amplitude, w_time, "{} wavelet".format(wavelet_type)


# #Testing wavelet contructor
# a_w_sampled, t_w_sampled, w_name = construct_wavelet(0.256, 0.001, 30, 'ricker')

# plt.figure(figsize=(8, 5))
# plt.plot(t_w_sampled, a_w_sampled)
# plt.xlabel('Time [ms]'); plt.ylabel('Amplitude'); plt.title(w_name.upper())
# plt.grid(True)
# plt.show()


#-----

#Time to frequency function

def time_to_freq(sampled_time, sampled_amplitude):
    """_summary_

    Args:
        sampled_time (array): wavelet time samples
        sampled_amplitude (array): wavelet amplitude samples
        
    Returns: 
        w_freq_cropped: Real component of the positive frequencies.
        w_ampl_cropped: Amplitudes corresponding to the positive frequencies.
        w_amp_spectrum: Amplitude spectrum based on w_ampl_cropped.
    """
    
    #time vector
    t_vector = sampled_time
    dt = t_vector[1]- t_vector[0]  #sampling rate
    num_samples_t = len(sampled_time)

    #frequency vector - nyquist frequency
    num_samples_f = int(pow(2, np.ceil(np.log(num_samples_t)/np.log(2))))
    df = 1 / (dt * num_samples_f)

    #freq sampling
    fmin = -0.5 * df * num_samples_f
    f_vector = fmin + df * np.arange(0, num_samples_f) #freq vector - allows to visualize the wave and it is linked to the time interval used for sampling 

    #wavelet in freq domain
    wave_amplitude = np.zeros(num_samples_f)
    wave_amplitude[:len(sampled_amplitude)] = sampled_amplitude

    #FOURIER TRANSFORM - Amplitude complete power
    wave_amplitude_f = np.fft.fft(wave_amplitude)
    wave_amplitude_fsh = np.fft.fftshift(wave_amplitude_f)

    #Amplitude positive power
    w_freq_cropped = f_vector[f_vector>=0]
    w_ampl_cropped = wave_amplitude_fsh[f_vector>=0]

    #Amplitud Spectum
    w_amp_spectrum = np.abs(w_ampl_cropped)
    #w_amp_spectrum = np.abs(w_ampl_cropped) / max(np.abs(w_ampl_cropped))
    
    return w_freq_cropped, w_ampl_cropped, w_amp_spectrum


# #Testing function
# #Wavelet contructor
# amp_test, tim_test, name_test = construct_wavelet(0.256, 0.001, 35, 'ricker')

# #Testing
# w_freq_cropped_test, w_ampl_cropped_test, w_amp_spectrum_test = time_to_freq(sampled_time=tim_test, sampled_amplitude=amp_test)

# #Plotting results to compare
# plt.figure(figsize=(12, 8))
# plt.plot(w_freq_cropped_test, w_ampl_cropped_test.real, color='black', label='Positive frequancy power')
# #plt.plot(w_freq_cropped_test, np.abs(w_ampl_cropped_test), color='black', label='Positive frequency spectrum')
# plt.plot(w_freq_cropped_test, w_amp_spectrum_test, color='red', label='Positive frequency spectrum')
# plt.xlabel('Frequency [Hz]', size=15); plt.ylabel('Amplitude', size=15), plt.title("{} CROPPED FREQUENCY POWER".format(name_test.upper()), size=15)
# plt.grid(True); plt.legend(fontsize=15), plt.xlim([0, 400])
# plt.show()