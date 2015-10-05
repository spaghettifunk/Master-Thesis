form Change formants
 	real F1_(Hz) 500.0
 	real F2_(Hz) 1500.0
 	real F3_(Hz) 2500.0
 	real F4_(Hz) 0.0 (=no change)
 	real F5_(Hz) 0.0 (=no change)
	boolean Retrieve_intensity_contour 1
	boolean Retrieve_EQ_curve 0
endform
if f1 < 0
	f1 = 0
endif
if f2 < 0
	f2 = 0
endif
if f3 < 0
	f3 = 0
endif
if f4 < 0
	f4 = 0
endif
if f5 < 0
	f5 = 0
endif
include batch.praat
procedure action
	s$ = selected$("Sound")
	execute workpre.praat
	wrk = selected("Sound")
	sr = Get sample rate
	execute extractvowels.praat
	voc = selected("Sound")
include maxformant.praat
	To Formant (burg)... 0.005 5 maxformant 0.025 50
	sf1 = Get mean... 1 0 0 Hertz
	sf2 = Get mean... 2 0 0 Hertz
	sf3 = Get mean... 3 0 0 Hertz
	sf4 = Get mean... 4 0 0 Hertz
	sf5 = Get mean... 5 0 0 Hertz
	plus voc
	Remove
	df1 = f1 - sf1
	df2 = f2 - sf2
	df3 = f3 - sf3
	df4 = f4 - sf4
	df5 = f5 - sf5
	if maxformant < 5000
		maxformant = 5000
	endif
	select wrk
	hf = Filter (pass Hann band)... maxformant 0 100
	select wrk
	samplingfrequency = maxformant * 2
	ss = Resample... samplingfrequency 10
	formant = To Formant (burg)... 0.005 5 maxformant 0.025 50
	lpc1 = To LPC... samplingfrequency
	plus ss
	source = Filter (inverse)
	select formant
	filtr = Copy... filtr
	if f1<>0
		Formula (frequencies)... if row = 1 then self + df1 else self fi
	endif
	if f2<>0
		Formula (frequencies)... if row = 2 then self + df2 else self fi
	endif
	if f3<>0
		Formula (frequencies)... if row = 3 then self + df3 else self fi
	endif
	if f4<>0
		Formula (frequencies)... if row = 4 then self + df4 else self fi
	endif
	if f5<>0
		Formula (frequencies)... if row = 5 then self + df5 else self fi
	endif
	lpc2 = To LPC... samplingfrequency
	plus source
	tmp1 = Filter... no
	tmp2 = Resample... sr 10
	Formula...  self+Object_'hf'[]
	if retrieve_EQ_curve
		plus wrk
		execute copyeq.praat
		tmp3 = selected("Sound")
	endif
	if retrieve_intensity_contour
		plus wrk
		execute copyintensity.praat
		tmp4 = selected("Sound")
	endif
	execute workpost.praat
	result = Rename... 's$'_changeformants
	select wrk
	plus hf
	plus ss
	plus formant
	plus lpc1
	plus source
	plus filtr
	plus lpc2
	plus tmp1
	plus tmp2
	if retrieve_EQ_curve
		plus tmp3
	endif
	if retrieve_intensity_contour
		plus tmp4
	endif
	Remove
	select result
endproc
