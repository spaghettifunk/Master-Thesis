form Create waveform
	positive Duration_(s) 2.0
	positive Sampling_frequency_(Hz) 44100
	positive Frequency_(Hz) 130.81
	choice Type: 1
		button Pulse train
		button Sawtooth
		button Silence
		button Sine
		button Square
		button Triangle
		button White noise
		button Pink noise
		button Brown noise
endform
if type = 1
	p = 1/frequency
	Create Sound from formula... Pulse_train'frequency'Hz 1 0 duration sampling_frequency if x mod p<1/sampling_frequency then 1 else 0 fi
endif
if type = 2
	p = 1/frequency
	Create Sound from formula... Sawtooth'frequency'Hz 1 0 duration sampling_frequency 2/p*((x+p/2) mod p-p/2)
endif
if type = 3
	Create Sound from formula... Silence 1 0 duration sampling_frequency 0
endif
if type = 4
	Create Sound from formula... Sine'frequency'Hz 1 0 duration sampling_frequency sin(2*pi*frequency*x)
endif
if type = 5
	p = 1/frequency
	Create Sound from formula... Square'frequency'Hz 1 0 duration sampling_frequency if x mod p<=p/2 then 1 else -1 fi
endif
if type = 6
	p = 1/frequency
	Create Sound from formula... Triangle'frequency'Hz 1 0 duration sampling_frequency (4/p*abs((x+3*p/4) mod p-p/2)-1)
endif
if type = 7
	Create Sound from formula... White_noise 1 0 duration sampling_frequency randomUniform(-1, 1)
endif
if type = 8
	white = Create Sound from formula... white 1 0 duration sampling_frequency randomUniform(-1, 1)
	b1 = Filter (de-emphasis)... 35
	Scale... 0.9
	select white
	b2 = Filter (de-emphasis)... 350
	Scale... 0.6
	select white
	b3 = Filter (de-emphasis)... 3500
	Scale... 0.9
	pinkNoise = Create Sound from formula... Pink_noise 1 0 duration sampling_frequency 0
	Formula...  self + Object_'b1'[] + Object_'b2'[] + Object_'b3'[]
	select white
	plus b1
	plus b2
	plus b3
	Remove
	select pinkNoise
endif
if type = 9
	Create Sound from formula... Brown_noise 1 0 duration sampling_frequency randomUniform(-1, 1)
	De-emphasize (in-line)... 5
endif
Scale... 0.9999
