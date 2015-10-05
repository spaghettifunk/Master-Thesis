form Vocoder
	positive Frequency_(Hz) 130.81
	choice Carrier_Waveform: 1
		button Pulse train
		button Sawtooth
		button Square
		button Triangle
		button White noise
endform
include batch.praat
procedure action
	s = selected("Sound")
	s$ = selected$("Sound")
	dur = Get total duration
	sr = Get sample rate
	execute createwaveform.praat 'dur' 'sr' 'frequency' 'carrier_Waveform$'
	carrier = selected("Sound")
	select s
	tmp = Copy... tmp
	plus carrier
	execute copyvocoder.praat
	result = Rename... 's$'_vocoder_'carrier_Waveform$'
	select carrier
	plus tmp
	Remove
	select result
endproc
