sel1 = selected("Sound",1)
s1$ = selected$("Sound",1)
sel2 = selected("Sound",2)
s1tmp = sel1
s2tmp = sel2
sr1 = 1/Object_'s1tmp'.dx
sr2 = 1/Object_'s2tmp'.dx
if sr1<>sr2
	select s1tmp
	resampled = Resample... sr2 50
else
	select s1tmp
endif
s1 = Copy... s1tmp
select s2tmp
s2$ = selected$("Sound")
s2 = Copy... wrk
execute fixdc.praat
dur = Get total duration
int = Get intensity (dB)
select s1
execute ireq.praat
pulse_target = selected("Sound")
select s2
sp2 = To Spectrum... yes
Formula... if row = 1 then 1/self[1, col] else self fi
sp3 = Cepstral smoothing... 100
Filter (pass Hann band)... 80 0 20
Filter (pass Hann band)... 0 20000 100
tmp1 = To Sound
samples = Get number of samples
tmp2 = Create Sound from formula... tmp2 1 0 0.05 sr2 0
Formula... if col > 'sr2'/40 and col <= 'sr2'/20 then Object_'tmp1'[col-('sr2'/40)] else Object_'tmp1'[col+('samples'-('sr2'/40))] fi
pulse_inv = Extract part... 0 0.05 Hanning 1 no
Scale... 0.9999
plus pulse_target
pulse_eq_tmp = Convolve
pulse_eq = Extract part... 0.025 0.075 Hanning 1 no
Scale... 0.9999
plus s2
result_tmp = Convolve
result = Extract part... 0.025 dur+0.025 rectangular 1 no
Rename... 's2$'_copyeq_'s1$'
Scale intensity... int
execute fixdc.praat
select s1
plus s2
plus pulse_target
plus sp2
plus sp3
plus tmp1
plus tmp2
plus pulse_inv
plus pulse_eq_tmp
plus pulse_eq
plus result_tmp
if sr1<>sr2
	plus resampled
endif
Remove
select result
