n_batch = numberOfSelected("Sound")
for i_batch to n_batch
	bsel'i_batch' = selected("Sound", i_batch)
endfor
for i_batch to n_batch
	select bsel'i_batch'
	call action
	new_batch'i_batch' = selected()
endfor
if n_batch >= 1
	select new_batch1
	for i_batch from 2 to n_batch
		plus new_batch'i_batch'
	endfor
endif
