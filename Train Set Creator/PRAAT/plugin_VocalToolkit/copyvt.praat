s1 = selected("Sound",1)
s1$ = selected$("Sound",1)
s2 = selected("Sound",2)
s2$ = selected$("Sound",2)
select s1
include vt.praat
vt1 = vt
select s2
include vt.praat
vt2 = vt
vtshift = (vt1+1)-vt2
vtshift = 'vtshift:2'
if vtshift <= 0
	vtshift = 0.05
endif
if vtshift >= 2
	vtshift = 1.95
endif
select s2
execute changevt.praat 'vtshift'
Rename... 's2$'_copyvt_'s1$'
