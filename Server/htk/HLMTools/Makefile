# ----------------------------------------------------------- 
#                                                             
#                          ___                                
#                       |_| | |_/   SPEECH                    
#                       | | | | \   RECOGNITION               
#                       =========   SOFTWARE                  
#                                                             
#                                                             
# ----------------------------------------------------------- 
#         Copyright: Cambridge University
#          1995-2005 Engineering Department
#                    http://htk.eng.cam.ac.uk
#                    http://mi.eng.cam.ac.uk
#                 
#   Use of this software is governed by a License Agreement   
#    ** See the file License for the Conditions of Use  **    
#    **     This banner notice must not be removed      **    
#                                                             
# ----------------------------------------------------------- 
# File: HLMTools/Makefile.  Generated from Makefile.in by configure.
# ----------------------------------------------------------- 

SHELL   =	/bin/sh
inc     = 	../HTKLib
srcdir = .
top_srcdir = ..

prefix = /usr/local
exec_prefix = ${prefix}
bindir = ${exec_prefix}/bin
sbindir = ${exec_prefix}/sbin
libexecdir = ${exec_prefix}/libexec
datadir = ${prefix}/share
sysconfdir = ${prefix}/etc
sharedstatedir = ${prefix}/com
localstatedir = ${prefix}/var
libdir = ${exec_prefix}/lib
infodir = ${prefix}/info
mandir = ${prefix}/man
includedir = ${prefix}/include
oldincludedir = /usr/include
hlib    = 	../HTKLib
llib    = 	../HLMLib
HLIBS   = 	$(hlib)/HTKLib.a $(llib)/HLMLib.a
CC      = 	gcc
CFLAGS  = 	-Wall -Wno-switch -I/opt/X11/include -I/usr/include/malloc -I$(hlib) -I$(llib) 
LDFLAGS = 	-L/opt/X11/lib $(HLIBS) -lm
INSTALL = 	/usr/bin/install -c
PROGS   =	Cluster HLMCopy LAdapt LBuild LFoF \
		LGCopy LGList LGPrep LLink LMerge \
		LNewMap LNorm LPlex LSubset 

all:  $(PROGS)

# build $(PROGS)
%: %.c $(HLIBS)
	if [ ! -d $(bindir) -a X_ = X_yes ] ; then mkdir -p $(bindir) ; fi
	$(CC) -o $@ $(CFLAGS) $^ $(LDFLAGS)
	if [ X_ = X_yes ] ; then $(INSTALL) -m 755 $@ $(bindir) ; fi

strip: $(PROGS)
	-strip $(PROGS)

clean:
	-rm -f *.o 

cleanup:
	-rm -f *.o $(PROGS) *.exe

distclean:
	-rm -f *.o $(PROGS) Makefile *.exe

install: mkinstalldir $(PROGS)
	for program in $(PROGS) ; do $(INSTALL) -m 755 $${program} $(bindir) ; done

mkinstalldir:
        if [ ! -d $(bindir) -a X_ = X_yes ] ; then mkdir -p $(bindir) ; fi

.PHONY: all strip clean cleanup distclean install mkinstalldir
