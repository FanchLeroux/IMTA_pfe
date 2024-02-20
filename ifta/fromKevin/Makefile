##############################################################################
#
#       Simplish Makefile for people like me who don't
#		understand complicated things.
#
#			Kevin Heggarty
##############################################################################

############### Choose the complier and the necessary flags. #################

CC=	gcc
CFLAGS =  -g -O3 -m64 -mcmodel=medium #

########### Tell the compiler where the X and XVIEW libraries are. ###########

XVIEWDIR= /usr/openwin
LIBDIR 	= $(XVIEWDIR)/include

###### Tell the complier which directories to search for include files. ######

INCLUDES = -I$(LIBDIR)

############# Tell the compiler which are the required libraries. ############
 
LIBS = -L$(LIBDIR) -lm -lfftw3



######################### Do the Linking #####################

fftwtest1d : fftwtest1d.o 
	${LINK.c}  -o $@ fftwtest1d.o -lfftw3 ${LIBS}
fftwtest : fftwtest.o 
	${LINK.c}  -o $@ fftwtest.o -lfftw3 ${LIBS}
iftafastbin : iftafastbin.o 
	${LINK.c}  -o $@ iftafastbin.o -lfftw3 ${LIBS}
iftafastail2D : iftafastail2D.o
	${LINK.c}  -o $@ iftafastail2D.o -lfftw3 ${LIBS} 
iftaail2D : iftaail2D.o
	${LINK.c}  -o $@ iftaail2D.o -lfftw3 ${LIBS} 
iftabinw : iftabinw.o 
	${LINK.c}  -o $@ iftabinw.o -lfftw3 ${LIBS}
iftaquant : iftaquant.o 
	${LINK.c}  -o $@ iftaquant.o -lfftw3 ${LIBS}
iftaquantwpara : iftaquantwpara.o 
	${LINK.c}  -o $@ iftaquantwpara.o -lfftw3_threads -lfftw3 -lpthread ${LIBS}
iftaquantwXY : iftaquantwXY.o 
	${LINK.c}  -o $@ iftaquantwXY.o -lfftw3 ${LIBS}
iftaquant-phasew : iftaquant-phasew.o 
	${LINK.c}  -o $@ iftaquant-phasew.o -lfftw3 ${LIBS}
iftaailbin1d : iftaailbin1d.o
	${LINK.c}  -o $@ iftaailbin1d.o -lfftw3 ${LIBS} 
iftafastbin-phasew : iftafastbin-phasew.o
	 ${LINK.c}  -o $@ iftafastbin-phasew.o -lfftw3 ${LIBS} 
iftafastbin-phase2w : iftafastbin-phase2w.o
	 ${LINK.c}  -o $@ iftafastbin-phase2w.o -lfftw3 ${LIBS} 
iftafresnelbinw : iftafresnelbinw.o
	 ${LINK.c}  -o $@ iftafresnelbinw.o -lfftw3 ${LIBS} 
iftafresnelbinillumw : iftafresnelbinillumw.o
	 ${LINK.c}  -o $@ iftafresnelbinillumw.o -lfftw3 ${LIBS} 
iftafresnelquant : iftafresnelquant.o
	 ${LINK.c}  -o $@ iftafresnelquant.o -lfftw3 ${LIBS} 
iftafresnelquantspher : iftafresnelquantspher.o
	 ${LINK.c}  -o $@ iftafresnelquantspher.o -lfftw3 ${LIBS} 
iftafastbinwXY : iftafastbinwXY.o 
	${LINK.c}  -o $@ iftafastbinwXY.o -lfftw3 ${LIBS}


######################### Do the compilation #####################

fftwtest1d.o:	fftwtest1d.c
	${CC} fftwtest1d.c $(INCLUDES) -c $(CFLAGS) -o fftwtest1d.o
fftwtest.o:	fftwtest.c
	${CC} fftwtest.c $(INCLUDES) -c $(CFLAGS) -o fftwtest.oi
iftafastbin.o:	iftafastbin.c
	${CC} iftafastbin.c $(INCLUDES) -c $(CFLAGS) -o iftafastbin.o
iftafastail2D.o:	iftafastail2D.c
	${CC} iftafastail2D.c $(INCLUDES) -c $(CFLAGS) -o iftafastail2D.o
iftaail2D.o:	iftaail2D.c
	${CC} iftaail2D.c $(INCLUDES) -c $(CFLAGS) -o iftaail2D.o
iftabinw.o:	iftabinw.c
	${CC} iftabinw.c $(INCLUDES) -c $(CFLAGS) -o iftabinw.o
iftaquant.o:	iftaquant.c
	${CC} iftaquant.c $(INCLUDES) -c $(CFLAGS) -o iftaquant.o
iftaquantwpara.o:	iftaquantwpara.c
	${CC} iftaquantwpara.c $(INCLUDES) -c $(CFLAGS) -o iftaquantwpara.o
iftaquantwXY.o:	iftaquantwXY.c
	${CC} iftaquantwXY.c $(INCLUDES) -c $(CFLAGS) -o iftaquantwXY.o
iftaquant-phasew.o:	iftaquant-phasew.c
	${CC} iftaquant-phasew.c $(INCLUDES) -c $(CFLAGS) -o iftaquant-phasew.o
iftaailbin1d.o:	iftaailbin1d.c
	${CC} iftaailbin1d.c $(INCLUDES) -c $(CFLAGS) -o iftaailbin1d.o
iftafastbin-phasew.o:	iftafastbin-phasew.c
	${CC} iftafastbin-phasew.c $(INCLUDES) -c $(CFLAGS) -o iftafastbin-phasew.o
iftafastbin-phase2w.o:	iftafastbin-phase2w.c
	${CC} iftafastbin-phase2w.c $(INCLUDES) -c $(CFLAGS) -o iftafastbin-phase2w.o
iftafresnelbinw.o:	iftafresnelbinw.c
	${CC} iftafresnelbinw.c $(INCLUDES) -c $(CFLAGS) -o iftafresnelbinw.o
iftafresnelbinillumw.o:	iftafresnelbinillumw.c
	${CC} iftafresnelbinillumw.c $(INCLUDES) -c $(CFLAGS) -o iftafresnelbinillumw.o
iftafresnelquant.o:	iftafresnelquant.c
	${CC} iftafresnelquant.c $(INCLUDES) -c $(CFLAGS) -o iftafresnelquant.o
iftafresnelquantspher.o:	iftafresnelquantspher.c
	${CC} iftafresnelquantspher.c $(INCLUDES) -c $(CFLAGS) -o iftafresnelquantspher.o
iftafastbinwXY.o:	iftafastbinwXY.c
	${CC} iftafastbinwXY.c $(INCLUDES) -c $(CFLAGS) -o iftafastbinwXY.o



#############################################################################
#
#				EXPLANATION
#
# The first line tells us that the executable "try" depends on the
# object files test.o,fft.o and fft.h. If any of these have been
# modified Make knows that to get an up to date "try", these files
# will have to be recompiled and linked. The second line is the link
# command, it links test.o, fft.o and the required libraries into an
# executable called try (-o $@).
#	The third line tells make that test.o depends on test.c and
# therefore that recompliation is necessary if test.c has changed.
# line four is the compilation command for test.c, it compiles it into
# test.o with the correct includes etc. Lines five and six are the
# corresponding dependancies and commands for fft2.c.
#
############################################################################



