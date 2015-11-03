s1tmp = selected("Sound",1)
s1$ = selected$("Sound",1)
s2tmp = selected("Sound",2)
s2$ = selected$("Sound",2)
select s1tmp
dur1 = Get total duration
select s2tmp
stt2 = Get start time
s2dur = Extract part... stt2 stt2+dur1 rectangular 1 no
select s1tmp
execute workpre.praat
s1tmp2 = selected("Sound")
Scale... 0.9999
execute gate.praat 40
s1 = selected("Sound")
select s2dur
execute workpre.praat
s2tmp2 = selected("Sound")
Scale... 0.9999
execute gate.praat 40
s2 = selected("Sound")
plus s1
sr1_or = 1/Object_'s1'.dx
sr2_or = 1/Object_'s2'.dx
if sr1_or <> sr2_or
	select s1
	ss1 = Resample... sr2_or 10
	ss2 = s2
else
	ss1 = selected("Sound",1)
	ss2 = selected("Sound",2)
endif
pred_order = round((1/Object_'ss1'.dx)/1000) + 2
select ss1
tdur = Get total duration
bgnoise = Create Sound from formula... bgnoise 1 0 'tdur' 'sr2_or' randomUniform(-1, 1)
Scale intensity... 0.01
select ss1
Formula...  self + Object_'bgnoise'[]
lpc1 = To LPC (burg)... pred_order 0.025 0.01 50
plus ss1
source = Filter (inverse)
select ss2
lpc2 = To LPC (burg)... pred_order 0.025 0.01 50
plus source
lpcsource = nowarn Filter... no
plus ss2
execute copyintensity.praat
tmp = selected("Sound")
execute workpost.praat
result = Rename... 's2$'_vocoder_'s1$'
select s2dur
plus s1tmp2
plus s1
plus s2tmp2
plus s2
plus bgnoise
plus lpc1
plus source
plus lpc2
plus lpcsource
if sr1_or <> sr2_or
	plus ss1
endif
plus tmp
Remove
select result
