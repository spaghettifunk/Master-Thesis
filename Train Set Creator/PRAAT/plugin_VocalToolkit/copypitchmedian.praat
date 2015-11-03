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
include minmaxf0.praat
minF0_1 = minF0
maxF0_1 = maxF0
To Pitch... 0 minF0_1 maxF0_1
f0_1 = Get quantile... 0 0 0.50 Hertz
Remove
select s2
include minmaxf0.praat
minF0_2 = minF0
maxF0_2 = maxF0
To Pitch... 0 minF0_2 maxF0_2
f0_2 = Get quantile... 0 0 0.50 Hertz
Remove
f0_f = f0_1/f0_2
select s2
dur = Get total duration
pitch = To Pitch... 0.01 minF0 maxF0
plus s2
manipulation = To Manipulation
pitchtier = Extract pitch tier
durationtier = Create DurationTier... 's2$' 0 dur
Add point... 0 1
plus manipulation
Replace duration tier
select pitchtier
Formula... self*f0_f
plus manipulation
Replace pitch tier
select manipulation
resynthesis = Get resynthesis (overlap-add)
execute workpost.praat
result = Rename... 's2$'_copypitchmedian_'s1$'
select s1
plus s2
plus pitch
plus manipulation
plus pitchtier
plus durationtier
plus resynthesis
Remove
select result
