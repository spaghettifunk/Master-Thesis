form Copy duration
	choice Method 1
		button Stretch
		button Cut/add time
endform
s1 = selected("Sound",1)
s1$ = selected$("Sound",1)
s2 = selected("Sound",2)
s2$ = selected$("Sound",2)
select s1
dur1 = Get total duration
select s2
dur2 = Get total duration
Copy... wrk
if dur1<>dur2
	wrk = selected("Sound")
	execute fixdc.praat
	if method = 1
include minmaxf0.praat
		Lengthen (overlap-add)... 'minF0' 'maxF0' 'dur1'/'dur2'
	elsif method = 2
		Extract part... 0 dur1 rectangular 1 no
	endif
	result = Rename... 's2$'_copyduration_'method$'_'s1$'
	select wrk
	Remove
	select result
	execute fixdc.praat
else
	Rename...  's2$'_copyduration_'method$'_'s1$'
endif
