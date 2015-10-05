form Mix
	real Mix_(0-100_%) 50
endform
if mix < 0
	mix = 0
endif
if mix > 100
	mix = 100
endif
amp1 = mix
amp2 = 100 - mix
s1 = selected("Sound",1)
s1$ = selected$("Sound",1)
s2 = selected("Sound",2)
s2$ = selected$("Sound",2)
select s1
wrk1 = Copy... wrk1
execute fixdc.praat
sr1 = Get sample rate
dur1 = Get total duration
nch1 = Get number of channels
select s2
wrk2 = Copy... wrk2
execute fixdc.praat
sr2 = Get sample rate
dur2 = Get total duration
nch2 = Get number of channels
if sr1 > sr2
	maxsr = sr1
else
	maxsr = sr2
endif
if dur1 > dur2
	maxdur = dur1
else
	maxdur = dur2
endif
if nch1 > nch2
	maxch = nch1
else
	maxch = nch2
endif
result = Create Sound from formula... 's2$'_mix_'s1$' 'maxch' 0 maxdur maxsr 0
Formula... self + 'amp1'*Object_'wrk1'()
Formula... self + 'amp2'*Object_'wrk2'()
Scale... 0.9999
select wrk1
plus wrk2
Remove
select result
