form EQ Preset
	boolean Prelisten_(press_Apply) 1
	positive Prelisten_time_(s) 5.0
	optionmenu Preset 3
include eqpresetslist.txt
endform
include batch.praat
procedure action
	s = selected("Sound")
	s$ = selected$("Sound")
	sr1 = Get sample rate
	eq_ir = Read from file... eq/'preset$'.Sound
	sr2 = Get sample rate
	if sr1<>sr2
		s1 = Resample... sr1 50
	else
		s1 = selected("Sound")
	endif
	select s
	wrk = Copy... wrk
	execute fixdc.praat
	if prelisten
		sordur = Get total duration
		if prelisten_time > sordur
			prelisten_time = sordur
		endif
		s2 = Extract part... 0 prelisten_time rectangular 1 no
	else
		s2 = selected("Sound")
	endif
	dur = Get total duration
	int = Get intensity (dB)
	Scale... 0.9999
	sp2 = To Spectrum... yes
	Formula... if row = 1 then 1/self[1, col] else self fi
	sp3 = Cepstral smoothing... 100
	Filter (pass Hann band)... 80 0 20
	Filter (pass Hann band)... 0 20000 100
	tmp1 = To Sound
	samples = Get number of samples
	tmp2 = Create Sound from formula... tmp2 1 0 0.05 sr1 0
	Formula... if col > 'sr1'/40 and col <= 'sr1'/20 then Object_'tmp1'[col-('sr1'/40)] else Object_'tmp1'[col+('samples'-('sr1'/40))] fi
	pulse_inv = Extract part... 0 0.05 Hanning 1 no
	Scale... 0.9999
	plus s1
	pulse_eq_tmp = Convolve
	pulse_eq = Extract part... 0.025 0.075 Hanning 1 no
	Scale... 0.9999
	plus s2
	result_tmp = Convolve
	result_tmp2 = Extract part... 0.025 dur+0.025 rectangular 1 no
	Scale intensity... int
	execute fixdc.praat
	finalint = Get intensity (dB)
	Scale... 0.9999
	execute limiter.praat 85 1
	Scale intensity... finalint
	execute declip.praat
	result = Rename... 's$'_EQPreset_'preset$'
	select eq_ir
	plus s1
	plus wrk
	plus s2
	plus sp2
	plus sp3
	plus tmp1
	plus tmp2
	plus pulse_inv
	plus pulse_eq_tmp
	plus pulse_eq
	plus result_tmp
	plus result_tmp2
	Remove
	select result
	if prelisten
		Play
		Remove
		select s
	endif
endproc
