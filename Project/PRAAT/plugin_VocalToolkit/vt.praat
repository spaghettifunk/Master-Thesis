selsnd_f = selected("Sound")
execute extractvowels.praat
voc = selected("Sound")
include maxformant.praat
To Formant (burg)... 0.005 5 maxformant 0.025 50
f1 = Get mean... 1 0 0 Hertz
plus voc
Remove
if f1 = undefined
	f1 = 0
endif
f1 = 'f1:2'
y = f1 / 500
y = y*10
a = 0.0327
b = 0.1976
c = 5.3303
vt = (-b+(sqrt(abs((b*b)-(4*a*(c-y))))))/(2*a)
vt = vt/10
vt = vt + 0.15
vt = round(vt * 20) / 20
if vt <= 0
	vt = 0.05
endif
if vt >= 2
	vt = 1.95
endif
select selsnd_f
