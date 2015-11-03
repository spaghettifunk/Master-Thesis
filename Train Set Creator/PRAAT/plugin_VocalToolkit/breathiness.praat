form Breathiness
	natural Breathiness_(1-100_%) 25
endform
if breathiness > 100
	breathiness = 100
endif
if breathiness <= 50
	gain1 = 1
	gain2 = breathiness/50
endif
if breathiness > 50
	gain1 = (100-breathiness)/50
	gain2 = 1
endif
include batch.praat
procedure action
	s$ = selected$("Sound")
	wrk = Copy... wrk
	execute fixdc.praat
	dur = Get total duration
	sr = Get sample rate
	execute whisper.praat
	whisper = selected("Sound")
	result = Create Sound... 's$'_breathiness_'breathiness' 0 dur sr 0
	Formula...  self + 'gain1'*Object_'wrk'[]
	Formula...  self + 'gain2'*Object_'whisper'[]
	execute fixdc.praat
	select wrk
	plus whisper
	Remove
	select result
endproc
