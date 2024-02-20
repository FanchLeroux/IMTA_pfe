/*****************************************************************************

				ITFAQUANT.C                      11/6/08

        Uses fftw3 routines.

	This program is an IFTA calculation of a multi-phaselevel CGH.
        The recentring is used on the final hologramme calculation to be
        compatible with the optical reconstruction (WYSIWYG ?).

	The input image can be with a flat phase or a random phase and
        placed anywhere in the input plane: on or off-axis The FFT is direct
        No multiple FFT oversampling) so oversampling can be performed
	by including a small image in a larger hologram (see OFFX and OFFY)..
	The output hologram is *.pgm the 255 levels corresponding to
	0 to 2*PI phase.

	Possible improvements:
        (1) "Fast" version by removal of imageFFT recentring - not clear as
        the (re,im) to (amp,pha) transformation must be done anyway and it
        appears more efficient to do it only once.
        (2) Optimisation of the variation of the percentage of amplitude and
        phase pixels quantified with the iteration number in the holoconstrain()
        routines. Current variation seems to give reasonable results in most
        cases but it is a manual adjustment. Systematic study required. Idea:
        statistical study to regularly increase the percentage quantified.
        (3) Uniformity optimisation by over compensation doesn't seem to work
        whereas it does with binary phase IFTA. To be investigated (25/10/21).

	INPUTS : phaselevels out.hol out.pgm seed

*****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <fftw3.h>
#define NX 17                     /* image size */
#define NY 17                      /* image size */
#define N 600                      /* holo  size */
#define PI M_PI
#define SIGIMA 1.0                 /* 1.0: no saturation, 0.5: sat=0.5*top */
#define LAMFAC 1.1               /* Target reinforcement factor 1.25 */
#define ZERO 1.0e-37
#define OFFX 292                 /* Image position offset */
#define OFFY 292                  /* Image position offset */
#define ITEROUT 2000              /* save image and holo if iter==ITEROUT */

fftw_complex holo[N][N];
float image[NY][NX];
double phasefull[2*N],phasehalf[2*N];
int iter,count,seed;
 
int main(argc,argv)

int argc;
char *argv[];

{       void getimage(),prefftcoding(),postfftcoding();
	void holoconstrain1(),holoconstrain2();
	void imageconstrain(),outhol(),outimage();
	fftw_plan pfor, pback;
	FILE *input_file, *output_file;
	double eta;
	int nel,nsens,phalev;
	char c;

	nel=N;iter=1;count=0;
	phalev=atoi(argv[1]);
	
	input_file = fopen("../pl-opt-br-002.wisdom", "r");
	//input_file = fopen("../pc-opt-195.wisdom", "r");
	//input_file = fopen("../pc-opt-199.wisdom", "r");
	if (!fftw_import_wisdom_from_file(input_file))
          printf("Error reading wisdom!\n");
	fclose(input_file); /* be sure to close the file! */

	pfor = fftw_plan_dft_2d(N, N, &holo[0][0], &holo[0][0], FFTW_FORWARD,
                            FFTW_PATIENT);
	pback = fftw_plan_dft_2d(N, N, &holo[0][0], &holo[0][0], FFTW_BACKWARD,
                            FFTW_PATIENT);

	output_file = fopen("../pl-opt-br-002.wisdom", "w");
	//output_file = fopen("../pc-opt-195.wisdom", "w");
	//output_file = fopen("../pc-opt-199.wisdom", "w");
	fftw_export_wisdom_to_file(output_file);
	fclose(output_file); /* be sure to close the file! */

        getimage("../targets/circle17.pgm");
	
   	printf("Calculating diffuser ... \n");     // recalculate
	do
	{      nsens=0;
	       prefftcoding(nsens,1);
	       fftw_execute(pfor);
	       postfftcoding(nsens,1);
	       holoconstrain1("testhol.pgm");
	       nsens=1;
	       prefftcoding(nsens,1);
	       fftw_execute(pback);
	       postfftcoding(nsens,1);
	       imageconstrain("testima.pgm");
	       iter++;count++;
	}while(count<50);

	eta=0.05;count=0;iter=2;
	printf("\n\nQuantizing hologram ... \n");
	do
	{      nsens=0;     
	       prefftcoding(nsens,1);
	       fftw_execute(pfor);
	       postfftcoding(nsens,1);
	       holoconstrain2(phalev,eta);

	       nsens=1;
	       prefftcoding(nsens,1);
	       fftw_execute(pback);
	       postfftcoding(nsens,1);
	       imageconstrain("testima.pgm");
	       if(count%5==0)
	       {   eta=(eta<0.4) ? eta+0.05 : eta+0.01;
	           count=0;
	       }
	       iter++;count++;
	}while(eta<0.5);

	printf("\n\nWriting final binary hologram ... \n");
        nsens=0;eta=0.5;
	prefftcoding(nsens,1);
	fftw_execute(pfor);
	postfftcoding(nsens,1);
	holoconstrain2(phalev,eta);
	outhol(argv[2],phalev);
	nsens=1;
       	prefftcoding(nsens,1);
	fftw_execute(pback);
	postfftcoding(nsens,1);

       	outimage(argv[3]);

	fftw_destroy_plan(pfor);
	fftw_destroy_plan(pback);
	//printf("\n");
}

/*****************************************************************************/
/*****************************************************************************/

void getimage(fileimage)

char *fileimage;
/* char *filephase; */

{       extern fftw_complex holo[N][N];
	extern float image[NY][NX];
	extern double phasefull[2*N],phasehalf[2*N];
	extern int seed;
	float temp[N];
	double amp;
	int nx,ny,i,j,dummy;
	unsigned char c;
	FILE *fp;

	if((fp=fopen(fileimage,"r"))==NULL){
	  printf("Error loading image file  ...Exiting\n");
	  exit(0);
	}

	c=fgetc(fp);c=fgetc(fp);               /* Get *.pgm file type */
	do
	{        while((c=fgetc(fp))!='\n');;
	}while((c=fgetc(fp))=='#');            /* Skip comment lines */
	ungetc(c,fp);                          /* Put back good char */
	dummy=fscanf(fp,"%d %d\n",&nx,&ny);    /* Get image dimensions */
	dummy=fscanf(fp,"%d",&i);          /* Get grey levels */
	fgetc(fp);                         /* Get ONE carriage return */
	printf("Input image : x=%d y=%d grey=%d\n",nx,ny,i);

	for(i=0;i<NY;i++)
	{   for(j=0;j<NX;j++)
	    {     image[i][j]=(fgetc(fp))/255.0;   /* Normalise 0-1 */
	    }	
	}
	fclose(fp);

	image[NY/2][NX/2]=0.5;   // Artificially lower zeroth order: also see sigmazero
	
/*	if((fp=fopen(filephase,"r"))==NULL){
	  printf("Error loading phase file  ...Exiting\n");
	  exit(0);
	}
*/
/*	fread(temp,sizeof(float),(N),fp);                 /* diffuser init */
/*	fread(holo,sizeof(complex),(N*N),fp);      /* Whole image diffuser */
/*	fclose(fp); */

	for(i=0;i<N;i++)
	{	for(j=0;j<N;j++)
		{     holo[i][j][0]=0.0;
		      holo[i][j][1]=0.0;
		}
	}

	srand48(seed);

	for(i=OFFY;i<OFFY+NY;i++)
	{	for(j=OFFX;j<OFFX+NX;j++)
		{     	holo[i][j][0]=sqrt(image[i-OFFY][j-OFFX]);
/*			holo[i][j][1]=temp[i]+temp[j];    /* diffuser init */
			holo[i][j][1]=2.0*PI*drand48();    /* random phase */
		}
	}
	
	for(i=0;i<2*N;i++)
	{       phasehalf[i]=(PI/N)*((i*(N-1))%(2*N));
	        phasefull[i]=PI*(i%2);
	}
}

/*****************************************************************************/

void prefftcoding(nsens,cent)

int nsens,cent;

{	extern fftw_complex holo[N][N];
	extern double phasefull[2*N],phasehalf[2*N];
	double amp,pha,*phase;
	int i,j;

	if(cent==1)
	  phase=&phasefull[0];  /* Full pixel centring: centre pixel N/2 */
	else
	  phase=&phasehalf[0];  /* Half pixel centring: centre pixel (N-1)/2 */

	if(nsens==0)
	{     for(i=0;i<N;i++)
	      {	    for(j=0;j<N;j++)
		    {     amp=holo[i][j][0];
			  pha=holo[i][j][1]+*(phase+i+j);
			  holo[i][j][0]=amp*cos(pha);
			  holo[i][j][1]=amp*sin(pha);
		    }
	      }
	}
	else
	{     for(i=0;i<N;i++)
	      {     for(j=0;j<N;j++)
	            {    amp=holo[i][j][0];
			 pha=holo[i][j][1]- *(phase+i+j);
			 holo[i][j][0]=amp*cos(pha);
			 holo[i][j][1]=amp*sin(pha);
		    }
      	      }
	}
}

/*****************************************************************************/

void postfftcoding(nsens,cent)

int nsens,cent;

{	extern fftw_complex holo[N][N];
	extern double phasefull[2*N],phasehalf[2*N];
	double x,y,amp,*phase;
	int i,j;

	if(cent==1)
	  phase=&phasefull[0];  /* Full pixel centring: centre pixel N/2 */
	else
	  phase=&phasehalf[0];  /* Half pixel centring: centre pixel (N-1)/2 */

	if(nsens==0)
	{     for(i=0;i<N;i++)
	      {	    for(j=0;j<N;j++)
		    {	  x=holo[i][j][0];y=holo[i][j][1];
		          amp=hypot(x,y);
			  if(amp>ZERO)
			      holo[i][j][1]=atan2(y,x)+ *(phase+i+j);
			  else
			  {   holo[i][j][1]= *(phase+i+j);
/*			      printf("TOOSMALL\n");          */
			  }
			  holo[i][j][0]=amp;
		    }
    	      }
        }
        else
	{     for(i=0;i<N;i++)
              {	    for(j=0;j<N;j++)
		    {	  x=holo[i][j][0];y=holo[i][j][1];
		          amp=hypot(x,y);
			  if(amp>ZERO)
			       holo[i][j][1]=atan2(y,x)- *(phase+i+j);
			  else
			  {   holo[i][j][1]= - *(phase+i+j);
/*			      printf("TOOSMALL\n");          */
			  }
			  holo[i][j][0]=amp;
		    }
  	      }
        }
}


/*****************************************************************************/
/*****************************************************************************/

void imageconstrain(fileimage)

char *fileimage;

{	extern fftw_complex holo[N][N];
	extern float image[NY][NX];
	extern int iter;
	double smasum,bigsum,sumsqu,bigtop,error,e,x,z,targave;
	double crossamp,crossint,lamamp,lamint,lamfact;
	unsigned char c;
	int i,j,sman;
	FILE *fp;

	sman=NX*NY;

	bigtop=0.0;                                   /* whole image stats */
	for(i=0;i<N;i++)
	{	for(j=0;j<N;j++)
		{	z=holo[i][j][0];
		        if(z > bigtop)
			  bigtop=z;
		}
	}

	bigsum=0.0;                              /* normalise whole image */
	for(i=0;i<N;i++)
	{	for(j=0;j<N;j++)
		{	z=holo[i][j][0]/bigtop;
		        holo[i][j][0]=z;
			z*=z;                                 /* intensity */
			bigsum+=z;
		}
	}

	if(iter%ITEROUT==0)
	{  fp=fopen(fileimage,"w+");
	   fprintf(fp,"P5\n# CREATOR: iftabin.c : K.Heggarty Dept Optique\n");
	   fprintf(fp,"%d %d\n",N,N);
	   fprintf(fp,"255\n");

	   for(i=0;i<N;i++)
	   {	 for(j=0;j<N;j++)
		 {     z=holo[i][j][0];
		       z*=z;                        /* write out intensity */
		       c=(char)(z*255);
		       fputc(c,fp); 
		 }
	   }
	   fclose(fp);
	}

	smasum=0.0;sumsqu=0.0;
	crossamp=0.0;crossint=0.0;                  /* target image  stats */
	for(i=OFFY;i<(OFFY+NY);i++)
	{	for(j=OFFX;j<OFFX+NX;j++)
		{	z=holo[i][j][0];
			x=image[i-OFFY][j-OFFX];        /* intensity image */
			crossamp+=(sqrt(x)*z);
			z*=z;
			smasum+=z;
			crossint+=(x*z);
			sumsqu+=(z*z);
		}
	}
 	lamamp=crossamp/smasum;lamint=crossint/sumsqu;targave=smasum/(NX*NY);

	error=0.0;                         /* normalised error calculation */
	for(i=OFFY;i<(OFFY+NY);i++)
	{	for(j=OFFX;j<OFFX+NX;j++)
		{	z=holo[i][j][0];
			z*=z;                           /* intensity image */
			e=image[i-OFFY][j-OFFX] - lamint*z;
			error+=(double)(e*e);
		}
        }

	lamfact=LAMFAC/lamamp;
	printf("targave=%f  lamfact=%f\n",targave,lamfact);
	
	for(i=OFFY;i<(OFFY+NY);i++)                   /* reset target area */
	{   for(j=OFFX;j<OFFX+NX;j++)
	    {   if( (image[i-OFFY][j-OFFX]) > (0.2000) )
	          { if(iter<570)                       // overcompensation after N iterations, eg 50
		      {  holo[i][j][0]=lamfact*sqrt(image[i-OFFY][j-OFFX]);		     
		      }
		    else
		      {  holo[i][j][0]=holo[i][j][0] + 0.05*(targave-holo[i][j][0]); 
		      }
		  }
	    }
	}

	printf("Iter=%d Lambda=%.4f ",iter,lamamp);
	printf("Error=%.2f\t ",255.0*sqrt((error/sman)));
	printf("Efficiency=%.2f\n",100.0*(smasum/bigsum));

}

/****************************************************************************/
/****************************************************************************/

void holoconstrain1(filehol)

char *filehol;

{	extern fftw_complex holo[N][N];
	double x,z,mean,squares,top,sigma;
	extern int iter;
	unsigned char c;
	int i,j;
	FILE *fp;

	mean=0.0;squares=0.0;top=0.0;
	
	for(i=0;i<N;i++)
	{	for(j=0;j<N;j++)
		{ 	z=holo[i][j][0];
			if(z>top)
			  top=z;
			mean+=z;squares+=z*z;
		}
	}

	if(iter%ITEROUT==0){
	fp=fopen(filehol,"w");
  	fprintf(fp,"P5\n# CREATOR: iftabin.c : K.Heggarty Dept Optique\n");
	fprintf(fp,"%d %d\n",N,N);
        fprintf(fp,"255\n");

	for(i=0;i<N;i++)
	{	for(j=0;j<N;j++)
		{	z=fabs(holo[i][j][0]);     /*  Write out amplitude */
			c=(z>top) ? 255 : (255*z/top);
			fputc(c,fp);
		}
	}
	fclose(fp);
	}

	mean/=(N*N);squares/=(N*N);
	sigma=sqrt(squares-(mean*mean));
	printf("Meanhol=%.3f  sighol=%.3f  ",mean,sigma);
	printf("tophol=%.3f\n",top);
	top=top/(1.2+12.0/iter);

      	for(i=0;i<N;i++)
	{	for(j=0;j<N;j++)
		{	if(holo[i][j][0]>top) 
			      holo[i][j][0]=1.0;
			else
 			      holo[i][j][0]/=top;
		}
	}
}

/*****************************************************************************/
/*****************************************************************************/

void holoconstrain2(phalev,eta)

int phalev;
double eta;


{	extern fftw_complex holo[N][N];
        double x,z,mean,squares,top,sigma,twopi,phanorm;
	extern int iter,count;
	int i,j,ampquant,phaquant;
	unsigned char c;
	FILE *fp;

	mean=0.0;squares=0.0;top=0.0;

	for(i=0;i<N;i++)
	{	for(j=0;j<N;j++)
		{   	z=holo[i][j][0];
			if(z>top)
			  top=z;
			mean+=z;squares+=z*z;
		}
	}

/*	fp=fopen("testhol.pgm","w");
  	fprintf(fp,"P5\n# CREATOR: iftabin.c : K.Heggarty Dept Optique\n");
	fprintf(fp,"%d %d\n",N,N);
        fprintf(fp,"255\n");

	for(i=0;i<N;i++)
	{	for(j=0;j<N;j++)
		{	z=fabs(holo[i][j][0]);     /*  Write out amplitude
			c=(z>top) ? 255 : (255*z/top);
			fputc(c,fp);
		}
	}
	fclose(fp);
*/
	
	mean/=(N*N);squares/=(N*N);
	sigma=sqrt(squares-(mean*mean));
	printf("Meanhol=%.3f  sighol=%.3f  ",mean,sigma);
	printf("tophol=%.3f  eta=%.4f\n",top,eta);
	top=top/(1.25+iter/100.0);
	twopi=2.0*PI;phanorm=twopi/phalev;

	ampquant=0;phaquant=0;

      	for(i=0;i<N;i++)
	{	for(j=0;j<N;j++)
		{	if(holo[i][j][0]>top)
		        {   holo[i][j][0]=1.0;
			    ampquant++;
			}
			else
			    holo[i][j][0]/=top;
		        x=holo[i][j][1]-twopi*floor(holo[i][j][1]/twopi);
      			x=x/phanorm;
		        z=floor(0.5+x);
			if((fabs(x-z))<=eta)
			{   holo[i][j][1]=z*phanorm;
			    phaquant++;
			}
		}
	}
	printf("aquant=%d  pquant=%d\n",ampquant,phaquant);
}

/*****************************************************************************/
/*****************************************************************************/

void outhol(filename,phalev)

char *filename;
int phalev;

{	extern fftw_complex holo[N][N];
	int i,j,d,hist[256];
        double phanorm1;
	unsigned char c;
	FILE *fp;

	phanorm1=phalev/(2.0*PI);

	fp=fopen(filename,"w");
  	fprintf(fp,"P5\n# CREATOR: iftabin.c : K.Heggarty Dept Optique\n");
	fprintf(fp,"%d %d\n",N,N);
        fprintf(fp,"255\n");

	for(i=0;i<256;i++)
	  hist[i]=0;


	for(i=0;i<N;i++)
	{   for(j=0;j<N;j++)
	    {	d=(int)(rint(phanorm1*(holo[i][j][1])));  // pha arrives 0-2PI
	        d=d%phalev;
	        //c=(int)rint(((d+drand48())*256.0)/phalev);         // noise on phase levels
		c=(int)rint((d*256.0)/phalev);
		//if(c==0)           // compensate for offset LUT
		//  c=1;
		fputc(c,fp);
		hist[c]=hist[c]+1;
		holo[i][j][0]=1.0;
		holo[i][j][1]=c*2.0*PI/256.0;
	    }
	}
	fclose(fp);

	for(i=0;i<phalev;i++)       /* Histogram of phase levels */
	{       j=(int)(floor(0.5+i*256.0/phalev));
/*		printf("hist[%d] = %d\n",j,hist[j]); */
	}


}

/*****************************************************************************/
/*****************************************************************************/

void outimage(filename)

char *filename;

{	extern fftw_complex holo[N][N];
	extern float image[NY][NX];
	double smasum,bigsum,sumsqu,cross,lambda,error,bigtop,x,z,e;
	unsigned char c;
	int i,j,bign,sman;
	FILE *fp;

	bign=N*N;sman=NX*NY;


	bigtop=0;bigsum=0.0;                          /* whole image stats */
	for(i=0;i<N;i++)
	{	for(j=0;j<N;j++)
		{	z=holo[i][j][0];
			z*=z;                                 /* Intensity */
			holo[i][j][0]=z;
			if(z > bigtop)
			  bigtop=z;
			bigsum+=z;
		}
	}

	smasum=0.0;sumsqu=0.0;cross=0.0;             /* target image stats */
	for(i=OFFY;i<(OFFY+NY);i++)
	{	for(j=OFFX;j<OFFX+NX;j++)
		{	z=holo[i][j][0];
			x=image[i-OFFY][j-OFFX];
			smasum+=z;
			sumsqu+=(z*z);
			cross+=x*z;
		}
	}

	lambda=cross/sumsqu;error=0.0;     /* normalised error calculation */
		        printf("cross=%f\n",cross);
		        printf("lambda=%f\n",lambda);

	for(i=OFFY;i<(OFFY+NY);i++)
	{	for(j=OFFX;j<OFFX+NX;j++)
		{	e=image[i-OFFY][j-OFFX] - lambda*holo[i][j][0];
			error+=(e*e);           /* MSE between intensities */
		}
        }

	printf("Error (RMS)=%.2f\t ",255.0*sqrt((error/sman))); 
	printf("Efficiency (%%)=%.2f\n",100.0*(smasum/bigsum));

	fp=fopen(filename,"w+");
  	fprintf(fp,"P5\n# CREATOR: iftabin.c : K.Heggarty Dept Optique\n");
	fprintf(fp,"%d %d\n",N,N);
        fprintf(fp,"255\n");
	for(i=0;i<N;i++)                  /* output whole image normalized */
	{	for(j=0;j<N;j++)
		{	z=holo[i][j][0]/bigtop;
		        holo[i][j][0]=z;
			c=(z>SIGIMA) ? 255 : (z*255/SIGIMA);
			fputc(c,fp);
		}
	}
	fclose(fp);
}
