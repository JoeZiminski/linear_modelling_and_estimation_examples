import numpy as np
from matplotlib import pyplot as plt
# make into jupiter notebook
# add more detail to extracting phase etc.
# write into blog?
# add modulation property 
# https://dspillustrations.com/pages/posts/misc/properties-of-the-fourier-transform.html

# ---------------------------------------------------------------------------------------
# A rotation in the Fourier Domain is a shift in the time domain.
# ---------------------------------------------------------------------------------------

t = 2*np.pi
x = np.linspace(0, t, 1000)
fs = x.size / t
n = x.size
print(t)

y1 = np.sin(x*5)
y2 = 2 * np.sin(x*10+np.pi/3)
y3 = 3 * np.cos(x*15+np.pi/6)
y4 = 1.5 * np.cos(x*20-np.pi/10)

y = y1 + y2 + y3 + y4

plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.plot(x, y4)
plt.show()

plt.plot(x, y)
plt.show()

Y = np.fft.fftshift(np.fft.fft(y))
freqs = np.fft.fftfreq(x.size, 1/fs)  # this outputs positive then negative
freqs = np.fft.fftshift(freqs)  # this does in order negative to positive

# the maximum detectable frequency is half the sampling rate. for each frequency, we have an imaginary
# number and because of complex conjugation we have 2x the frequencies (positive and negative). We can
# ignore the negative frequencies. See below example, the binning algo of fftshift
# is a little tricky to conceptualise but below it can be seen it is essentially
# the fs/n * 0 ... n/2 (with some complication for odd numbers).

example_for_even_numbers = [i * fs/(n) for i in range(0, int((n)/2))]
example_for_even_numbers_2 = np.linspace(-fs/2, fs/2, n + 2)

# Normalisation - to return to actual units, we combine
# a + bi using abs (sqrt(a^2 + b^2)) - if we use Y.real this does not work!
# then divide by n as we have taken the integral, then multiply by 2 because
# we ignore the imaginary half of the spectrum.

norm_Y = (np.abs(Y) / n) * 2
norm_freqs = freqs * 2 * np.pi  # convert to Hz (i.e. the 'recording' lasts 2 * pi seconds)

plt.plot(norm_freqs, np.abs(Y))
plt.show()

# Here the DC term is almost zero because of the average of periodic sines / cosines is zero.
# However it is important to understand, , see below:
# https://dsp.stackexchange.com/questions/12972/discrete-fourier-transform-what-is-the-dc-term-really

# The DC term is the 0 Hz term and is equivalent to the average of all the samples# in the window
# hence it's always purely real for a real signal). The terminology does indeed come from AC/DC electricity
# -all the non-zero bins correspond to non-zero frequencies, i.e. "AC components" in an electrical context,
# whereas the zero bin corresponds to a fixed value, the mean of the signal, or "DC component" in electrical terms.

# We can see that when f = 0 in the fourier transform itnergral over x(t)*e^-i2pi*f*n/N then
# this reduces to integral over x(t) which I guess is why its the mean after normalisation (/ N)

# As far as practical applications go, the DC or 0 Hz term is not particularly useful.
# In many cases it will be close to zero, as most signal processing applications will
# tend to filter out any DC component at the analogue level.
# In cases where you might be interested it can be calculated directly as an average in
# the usual way, without resorting to a DFT/FFT.

# Plotting the phase term

plt.plot(norm_freqs, np.angle(Y))
plt.plot(norm_freqs, norm_Y)
plt.show()

#The complex representation of frequency is such that the real part corresponds
# to a cosine component and the imaginary part to a sine component.
# So a complex phase of 0 corresponds to a cosine wave, not a sine wave.
# This is why the computed phases are off by about 90 degrees from what you expect,
# according to the trig identity sin(x) = cos(x − π/2). The lower-frequency wave should have a
# phase of −90 degrees and the higher-frequency wave should have a phase of 30 − 90 = −60 degrees.

# So the phase is measured from cos(0) not sin(0). We can see from the plot (estimated):
# y1 [np.sin(x*5)]                 MAG = 1, FREQ = 5, PHASE OFFSET FROM COS(0) = 0 + 90 degree = pi / 2 = 1.5707
# y2 [2 * np.sin(x*10+np.pi/3)]    MAG = 2, FREQ = 10, PHASE OFFSET FROM COS(0) = pi/3 + 90 degree = 2.617
# y3 [3 * np.cos(x*15+np.pi/6)]    MAG = 3, FREQ = 15, PHASE OFFSET FROM COS(0) = pi / 6 = 0.523
# y4 [1.5 * np.cos(x*20-np.pi/10)] MAG = 1.5, FREQ = 20, PHASE OFFSET FROM COS(0) = pi / 10 = 0.314

# So we see that the norm_Y gives the power (i.e. size of deviation from the origin) and the
# np.angle gives the phase of the sine wave at the specified frequency (angle from positive x).

# Now, apply a rotation by multiplying the fourier transform by exponential with shift linearly
# proportional to the frequency. The change must be linearly proprotional to frequency rather
# than the same for all frequencies, because e.g. a half-period of a fast sine wave is shorter
# than a slow sine wave.
Y_rot = Y * np.exp(-1j * 2 * np.pi * freqs * np.pi/2)
plt.plot(norm_freqs, np.angle(Y_rot))
plt.plot(norm_freqs, (np.abs(Y_rot)/ n ) * 2)
plt.show()

# See, the shift has been applied! amazing!! beautiful!
y_rot = np.real(np.fft.ifft(np.fft.ifftshift(Y_rot)))
plt.plot(x, y)
plt.plot(x, y_rot)
plt.show()


# ---------------------------------------------------------------------------------------
# Cross Correlation in the Foruier Domain
# ---------------------------------------------------------------------------------------
