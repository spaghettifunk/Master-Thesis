form Fade
	real Fade_in_(s) 0.5
	real Fade_out_(s) 0.0 (=no change)
endform
if fade_in < 0
	fade_in = 0
endif
if fade_out < 0
	fade_out = 0
endif
include batch.praat
procedure action
	s$ = selected$("Sound")
	dur = Get total duration
	ns = Get number of samples
	t = fade_in
	if t < 0
		t = 0
	endif
	if t > dur
		t = dur
	endif
	s = t/ns
	Copy... 's$'_fade
	if t > 0
		Formula... if x<t then self*(1-cos(pi*(x/t)/(s-1)))/2 else self fi
	endif
	t = fade_out
	if t < 0
		t = 0
	endif
	if t > dur
		t = dur
	endif
	s = t/ns
	if t > 0
		Formula... if x>dur-t then self*(1-cos(pi*((dur-x)/t)/(s-1)))/2 else self fi
	endif
endproc
