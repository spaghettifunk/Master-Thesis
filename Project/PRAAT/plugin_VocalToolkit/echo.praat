form Echo
	boolean Prelisten_(press_Apply) 1
	positive Prelisten_time_(s) 3.0
	positive Delay_(s) 0.5
	positive Amplitude_(0.1-0.9) 0.5
endform
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
	if amplitude > 0.9
		amplitude = 0.9
	endif
	sr = Get sample rate
	dur = Get total duration
	tt= dur+(delay*(amplitude*2))*10
	tail = tt-dur
	tail = 'tail:2'
	result = Extract part... 0 tt rectangular 1 no
	Rename... 's$'_echo
	Formula... self+amplitude*self(x-delay)
	ns = Get number of samples
	ss = tail/ns
	Formula... if (x>(tt-tail)) then self*((tt-x)/tail) else self fi
	execute declip.praat
	select wrk
	Remove
	select result
	if prelisten
		Play
		plus pre
		Remove
		select s
	endif
endproc
