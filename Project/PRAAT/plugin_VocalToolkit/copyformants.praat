s1tmp = selected("Sound",1)
s1$ = selected$("Sound",1)
s2tmp = selected("Sound",2)
s2$ = selected$("Sound",2)
select s1tmp
execute workpre.praat
s1 = selected("Sound")
select s2tmp
execute workpre.praat
s2 = selected("Sound")
select s1
execute extractvowels.praat
vow1 = selected("Sound")
select s2
execute extractvowels.praat
vow2 = selected("Sound")
select vow1
include maxformant.praat
maxformant1 = maxformant
To Formant (burg)... 0.005 5 maxformant1 0.025 50
s1f1 = Get mean... 1 0 0 Hertz
s1f2 = Get mean... 2 0 0 Hertz
s1f3 = Get mean... 3 0 0 Hertz
s1f4 = Get mean... 4 0 0 Hertz
s1f5 = Get mean... 5 0 0 Hertz
plus vow1
Remove
select vow2
include maxformant.praat
maxformant2 = maxformant
To Formant (burg)... 0.005 5 maxformant2 0.025 50
s2f1 = Get mean... 1 0 0 Hertz
s2f2 = Get mean... 2 0 0 Hertz
s2f3 = Get mean... 3 0 0 Hertz
s2f4 = Get mean... 4 0 0 Hertz
s2f5 = Get mean... 5 0 0 Hertz
plus vow2
Remove
if s1f1 = undefined
	s1f1 = 0
endif
if s1f2 = undefined
	s1f2 = 0
endif
if s1f3 = undefined
	s1f3 = 0
endif
if s1f4 = undefined
	s1f4 = 0
endif
if s1f5 = undefined
	s1f5 = 0
endif
if s2f1 = undefined
	s2f1 = 0
endif
if s2f2 = undefined
	s2f2 = 0
endif
if s2f3 = undefined
	s2f3 = 0
endif
if s2f4 = undefined
	s2f4 = 0
endif
if s2f5 = undefined
	s2f5 = 0
endif
df1 = s1f1 - s2f1
df2 = s1f2 - s2f2
df3 = s1f3 - s2f3
df4 = s1f4 - s2f4
df5 = s1f5 - s2f5
if df1 > 300
	df1 = 300
endif
if df1 < -300
	df1 = -300
endif
if df2 > 300
	df2 = 300
endif
if df2 < -300
	df2 = -300
endif
if df3 > 300
	df3 = 300
endif
if df3 < -300
	df3 = -300
endif
if df4 > 300
	df4 = 300
endif
if df4 < -300
	df4 = -300
endif
if df5 > 300
	df5 = 300
endif
if df5 < -300
	df5 = -300
endif
if maxformant2 < 5000
	maxformant2 = 5000
endif
select s2
sr2 = Get sample rate
hf2 = Filter (pass Hann band)... maxformant2 0 100
select s2
samplingfrequency = maxformant2 * 2
ss2 = Resample... samplingfrequency 10
formant2 = To Formant (burg)... 0.005 5 maxformant2 0.025 50
lpc1 = To LPC... samplingfrequency
plus ss2
source = Filter (inverse)
select formant2
filtr = Copy... filtr
nocheck Formula (frequencies)... if row = 1 then self + df1 else self fi
nocheck Formula (frequencies)... if row = 2 then self + df2 else self fi
nocheck Formula (frequencies)... if row = 3 then self + df3 else self fi
nocheck Formula (frequencies)... if row = 4 then self + df4 else self fi
nocheck Formula (frequencies)... if row = 5 then self + df5 else self fi
lpc2 = To LPC... samplingfrequency
plus source
tmp = Filter... no
tmp2 = Resample... sr2 10
Formula...  self+Object_'hf2'[]
execute workpost.praat
result = Rename... 's2$'_copyformants_'s1$'
select s1
plus s2
plus hf2
plus ss2
plus formant2
plus lpc1
plus source
plus filtr
plus lpc2
plus tmp
plus tmp2
Remove
select result
