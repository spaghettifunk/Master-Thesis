selsnd_mf = selected("Sound")
To Formant (burg)... 0.005 6 5000 0.025 50
f1_pass1 = Get quantile... 1 0 0 Hertz 0.50
if f1_pass1 = undefined
	f1_pass1 = 0
endif
Remove
maxf_pass1 = round(f1_pass1/50)*500
select selsnd_mf
To Formant (burg)... 0.005 6 maxf_pass1 0.025 50
f1_pass2 = Get quantile... 1 0 0 Hertz 0.50
if f1_pass2 = undefined
	f1_pass2 = 0
endif
Remove
maxformant = round(f1_pass2/50)*500
select selsnd_mf
