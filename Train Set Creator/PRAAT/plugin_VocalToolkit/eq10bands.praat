form EQ 10 Bands
	boolean Prelisten_(press_Apply) 1
	positive Prelisten_time_(s) 3.0
	comment                                                                          (-24dB ... +24dB)
	integer Band_1_(31.5_Hz) -24
	integer Band_2_(63_Hz) -24
	integer Band_3_(125_Hz) -24
	integer Band_4_(250_Hz) -24
	integer Band_5_(500_Hz) 24
	integer Band_6_(1000_Hz) 24
	integer Band_7_(2000_Hz) 24
	integer Band_8_(4000_Hz) -24
	integer Band_9_(8000_Hz) -24
	integer Band_10_(16000_Hz) -24
endform
if band_1 < -24 or band_1 > 24 or band_2 < -24 or band_2 > 24 or band_3 < -24 or band_3 > 24 or band_4 < -24 or band_4 > 24 or band_5 < -24 or band_5 > 24 or band_6 < -24 or band_6 > 24 or band_7 < -24 or band_7 > 24 or band_8 < -24 or band_8 > 24 or band_9 < -24 or band_9 > 24 or band_10 < -24 or band_10 > 24
	exit Please enter values between -24 to 24
endif
include batch.praat
procedure action
	s = selected("Sound")
	s$ = selected$("Sound")
	wrk = Copy... wrk
	execute fixdc.praat
	if prelisten
		sordur = Get total duration
		if prelisten_time > sordur
			prelisten_time = sordur
		endif
		pre = Extract part... 0 prelisten_time rectangular 1 no
	endif
	sr = Get sample rate
	dur = Get total duration
	hifreq = sr/2
	pointprocess = Create empty PointProcess... pulse 0 0.05
	Add point... 0.025
	pulse = To Sound (pulse train)... sr 1 0.05 2000
	sp_pulse = To Spectrum... no
	buffer = Copy... buffer
	Formula... 0
	sp_eq = Copy... sp_eq
	call EqBand 0 44.2 band_1 20
	call EqBand 44.2 88.4 band_2 20
	call EqBand 88.4 177 band_3 40
	call EqBand 177 354 band_4 80
	call EqBand 354 707 band_5 100
	call EqBand 707 1414 band_6 100
	call EqBand 1414 2828 band_7 100
	call EqBand 2828 5657 band_8 100
	call EqBand 5657 11314 band_9 100
	call EqBand 11314 hifreq band_10 100
	Filter (pass Hann band)... 80 0 20
	Filter (pass Hann band)... 0 20000 100
	pulseeqtmp = To Sound
	pulsedur = Get total duration
	pulseeq = Extract part... (pulsedur-0.05)/2 pulsedur-((pulsedur-0.05)/2) Hanning 1 no
	Scale... 0.9999
	pulsesr = Get sample rate
	if pulsesr <> sr
		Override sampling frequency... sr
	endif
	plus wrk
	resulttmp = Convolve
	result = Extract part... 0.025 dur+0.025 rectangular 1 no
	Rename... 's$'_EQ10bands
	Scale... 0.9999
	select wrk
	plus pointprocess
	plus pulse
	plus sp_pulse
	plus buffer
	plus sp_eq
	plus pulseeqtmp
	plus pulseeq
	plus resulttmp
	Remove
	select result
	if prelisten
		Play
		plus pre
		Remove
		select s
	endif
endproc
procedure EqBand bnd1 bnd2 dB smoothing
	factor = 10^(dB/20)
	select buffer
	Formula... Object_'sp_pulse'[]
	Filter (pass Hann band)... bnd1 bnd2 smoothing
	Formula... self*factor
	select sp_eq
	Formula... self + Object_'buffer'[]
endproc
