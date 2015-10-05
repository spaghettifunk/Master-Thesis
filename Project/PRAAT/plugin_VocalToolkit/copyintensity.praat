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
pow1 = Get power... 0 0
intensityTarget = To Intensity... 100 0.01 0
intensityTierTarget = Down to IntensityTier
select s2
intensitySource = To Intensity... 100 0.01 0
max1 = Get maximum... 0 0 Parabolic
Formula... 'max1' - self
intensityTierSource = Down to IntensityTier
select s2
plus intensityTierSource
zero = Multiply
plus intensityTierTarget
tmp = Multiply
pow2 = Get power... 0 0
max2 = Get maximum... 0 0 None
nocheck Scale...  max2*sqrt(pow1/pow2)
execute workpost.praat
result = Rename... 's2$'_copyintensity_'s1$'
select s1
plus s2
plus intensityTarget
plus intensityTierTarget
plus intensitySource
plus intensityTierSource
plus zero
plus tmp
Remove
select result
