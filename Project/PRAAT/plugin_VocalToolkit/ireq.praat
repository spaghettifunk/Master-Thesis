sr = Get sample rate
sp = To Spectrum... yes
sp2 = Cepstral smoothing... 100
tmp1 = To Sound
samples = Get number of samples
tmp2 = Create Sound from formula... pulse 1 0 0.05 sr 0
Formula... if col > 'sr'/40 and col <= 'sr'/20 then Object_'tmp1'[col-('sr'/40)] else Object_'tmp1'[col+('samples'-('sr'/40))] fi
ireq = Extract part... 0 0.05 Hanning 1 no
Scale... 0.9999
select sp
plus sp2
plus tmp1
plus tmp2
Remove
select ireq
