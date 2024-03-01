 /*****************************************************************************

				ITFAFASTBIN.C

        Uses fftw3 routines.

	This program is a fast IFTA calculation of a binary off-axis CGH.
	It is basically the same program as iftabin.c only the recentring
	of the FFT is not performed during the diffuseur calculation nor
	the gradual binarization proceedure. The recentring is used on the
	final hologramme calculation to be compatible with the optical
	reconstruction (WYSIWYG ?).

	This program calculates an off-axis hologram. The input image
	can be with a flat phase or a random phase. The FFT is direct
	(no multiple FFT oversampling) so oversampling can be performed
	by including a small image in a larger hologram (see OFFX and OFFY)..
	The output hologram is *.pgm the 255 levels corresponding to
	0 to 2*PI phase.

	(Improvements: (1) removal of imageFFT recentring - not clear as
	the (re,im) to (amp,pha) transformation must be done anyway and
	it appears more efficient to do it only once. (2) Check recentring
	of final output hologramme, at the moment preparing a non-centred
	CGH then centring only the last iteration seems OK but it should
	be tested especially when the centring is not whole pixel as then
	we are not simply re-ordering the hologram pixels, the sampling
	point also changes so the results will not be the same. As the
	program is at the moment, the output and usage (eg. with OFFX,OFFY)
	corresponds exactly to that of iftabin.c)

	INPUTS : out.hol out.pgm seed

*****************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <fftw3.h>
#define NX 7                     /* image size */
#define NY 7                      /* image size */
#define N 2048                     /* holo  size */
#define PI M_PI
#define SIGIMA 1.0                 /* 1.0: no saturation, 0.5: sat=0.5*top */
#define LAMFAC 1.05                /* Target reinforcement factor: typ=1.2, small holo ~1.05*/
#define ZERO 1.0e-37
#define OFFX 1021                  /* Image position offset */
#define OFFY 1021   //568              /* Image position offset */
#define ITEROUT 2000              /* save image and holo if iter==ITEROUT */

fftw_complex holo[N][N];
float image[NY][NX];
double phasefull[2*N],phasehalf[2*N];
int iter,count,seed;
 
int main(argc,argv)

int argc;
char *argv[];

{       void getimage(),prefftcoding(),postfftcoding();
	void holoconstrain1(),holoconstrain2(),holoconstrain3();
	void imageconstrain(),outhol(),outimage();
	fftw_plan pfor, pback;
	FILE *input_file, *output_file;
	double eta;
	int nel,nsens;
	char c;

	nel=N;iter=1;count=0;
	
	//input_file = fopen("../sample.wisdom", "r");
	input_file = fopen("../pl-opt-br-002.wisdom", "r");
	//input_file = fopen("../srv-df-893.wisdom", "r");
	if (!fftw_import_wisdom_from_file(input_file))
	  printf("Error reading wisdom!\n");
	fclose(input_file); /* be sure to close the file! */
	
	pfor = fftw_plan_dft_2d(N, N, &holo[0][0], &holo[0][0], FFTW_FORWARD,
                            FFTW_PATIENT);
	pback = fftw_plan_dft_2d(N, N, &holo[0][0], &holo[0][0], FFTW_BACKWARD,
                            FFTW_PATIENT);
	

	output_file = fopen("../pl-opt-br-002.wisdom", "w");
	//output_file = fopen("../srv-df-893.wisdom", "w");
	fftw_export_wisdom_to_file(output_file);
	fclose(output_file); /* be sure to close the file! */

	seed=atoi(argv[3]);
        getimage("../targets/circle7.pgm");
	//getimage("../targets/cross23-1.pgm");
	//getimage("../images/IUT-153x152.pgm");

	//printf("Calculating diffuser ... \n");     /* recalculate */
	do
	{      nsens=0;
	       prefftcoding(nsens,1);
	       fftw_execute(pfor);
	       holoconstrain1("testhol.pgm");
	       nsens=1;
	       fftw_execute(pback);
	       postfftcoding(nsens,1);
	       imageconstrain("testima.pgm");
	       iter++;count++;
	}while(count<250);                           // standard=50 (100,200 ... )? 

	eta=1.0;count=0;iter=2;
	//printf("\n\nBinarizing hologram ... \n");
	do
	{      nsens=0;     
	       prefftcoding(nsens,1);
	       fftw_execute(pfor);
	       holoconstrain2(eta);
	       nsens=1;
	       fftw_execute(pback);
	       postfftcoding(nsens,1);
	       imageconstrain("testima.pgm");
	       if(count%5==0)          // Typical= 5, increase for small holo to improve uniformity
	       {   eta=(eta>0.2) ? eta-0.1 : eta-eta/2.0;
	           count=0;
	       }
	       iter++;count++;
	}while(eta>=0.01);                    // Decrease for small holo to improve uniformity

	//printf("\n\nWriting final binary hologram ... \n");
	nsens=0;eta=0.0;
	prefftcoding(nsens,1);
	fftw_execute(pfor);
	postfftcoding(nsens,1);
	holoconstrain3(eta);
	outhol(argv[1]);
	nsens=1;
       	prefftcoding(nsens,1);
	fftw_execute(pback);
	postfftcoding(nsens,1);

       	outimage(argv[2]);

	fftw_destroy_plan(pfor);
	fftw_destroy_plan(pback);
	//printf("\n\n");
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
	float x,y,z,r;
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
	dummy=fscanf(fp,"%d %d\n",&nx,&ny);          /* Get image dimensions */
	dummy=fscanf(fp,"%d",&i);                  /* Get grey levels */
	fgetc(fp);                         /* Get ONE carriage return */
	//printf("Input image : x=%d y=%d grey=%d\n",nx,ny,i);

	r=sqrt(NY*NY + NX*NX);

	for(i=0;i<NY;i++)
	{   for(j=0;j<NX;j++)
	    {     image[i][j]=(fgetc(fp))/255.0;   /* Normalise 0-1 */
	          /*image[i][j]=0.25+0.75*(fgetc(fp))/255.0;   /* Normalise 0-1 *	      
	          x=j-NX/2;y=i-NY/2;
	          z=sqrt(x*x + y*y);
 		  image[i][j]*=0.65+0.35*(z/r);    // rolloff */
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

	printf("seed=%d  ",seed);
	srand48(seed);

	for(i=OFFY;i<OFFY+NY;i++)
	{  for(j=OFFX;j<OFFX+NX;j++)
	   {  holo[i][j][0]=sqrt(image[i-OFFY][j-OFFX]);
	      holo[i][j][1]=2.0*PI*drand48();    /* random phase */
	      //holo[i][j][1]=temp[i]+temp[j];    /* diffuser init */
	      //holo[i][j][1]+=(0.4*PI/N)*((i-N/2)*(i-N/2)+(j-N/2)*(j-N/2)); /* spherique */
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
			  amp=sqrt(x*x+y*y);
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
			  amp=sqrt(x*x+y*y);
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

/****************************************************************************/

void holoconstrain1(filehol)

char *filehol;

{	extern fftw_complex holo[N][N];
	double x,z,mean,squares,top,sigma,limit,wfact;
	extern int iter;
	unsigned char c;
	int i,j;
	FILE *fp;

	mean=0.0;squares=0.0;top=0.0;
	
	for(i=0;i<N;i++)
	{	for(j=0;j<N;j++)
	        {   x=holo[i][j][0];             // Real Holo
		    z=fabs(x);
		    holo[i][j][1]=0.0;
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

	wfact=squares/(top*top*N*N);
	mean/=(N*N);squares/=(N*N);
	sigma=sqrt(squares-(mean*mean));
	limit=mean+sigma;
//	printf("Meanhol=%.3f  sighol=%.3f  ",mean,sigma);
//      printf("tophol=%.3f  Wfact=%f\n",top,wfact);
	top=top/(1.2+10.0/iter);
/*	top=top/(1.1); */

      	for(i=0;i<N;i++)
	{	for(j=0;j<N;j++)
		{	z=holo[i][j][0];
			if(fabs(z)>top) 
			      x=(z>0.0) ? 1.0 : -1.0;
			else
			      x=z/top;
			holo[i][j][0]=x;
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
	double smasum,bigsum,sumsqu,bigtop,error,e,x,z;
	double crossamp,crossint,lamamp,lamint,lamfact,targave;
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

	for(i=OFFY;i<(OFFY+NY);i++)                   /* reset target area */
	{   for(j=OFFX;j<OFFX+NX;j++)
	    {   if( (image[i-OFFY][j-OFFX]) > (0.001) )      // only change ON pixels
	        { if(iter<2000)                                 // overcompensation after N iterations
		  {  holo[i][j][0]=lamfact*sqrt(image[i-OFFY][j-OFFX]);
		     holo[N-i][N-j][0]=lamfact*sqrt(image[i-OFFY][j-OFFX]);
		  }
		  else
		  {  holo[i][j][0]=holo[i][j][0] + lamfact*(targave-holo[i][j][0]); 
		     holo[N-i][N-j][0]=holo[N-i][N-j][0] + lamfact*(targave-holo[N-i][N-j][0]);
		  }
		}
	    }
	}
	//printf("Iter=%d Lambda=%.4f ",iter,lamamp);
	//printf("Error=%.2f\t ",255.0*sqrt((error/sman)));
	//printf("Efficiency=%.2f\n",100.0*(smasum/bigsum));
}

/*****************************************************************************/
/*****************************************************************************/

void holoconstrain2(eta)

double eta;

{	extern fftw_complex holo[N][N];
	double x,z,mean,squares,top,sigma,limit,wfact,twopi;
	extern int iter,count;
	int i,j;
	unsigned char c;
	FILE *fp;

	mean=0.0;squares=0.0;top=0.0;

	for(i=0;i<N;i++)
	{	for(j=0;j<N;j++)
	        {       z=holo[i][j][0];   /* real holo */ 
			holo[i][j][1]=0.0;
			z=fabs(z);
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
	wfact=squares/(top*top*N*N);
	mean/=(N*N);squares/=(N*N);
	sigma=sqrt(squares-(mean*mean));
	limit=mean+sigma;
	//	printf("Meanhol=%.3f  sighol=%.3f  ",mean,sigma);
/*	printf("tophol=%.3f  Wfact=%f  eta=%.4f\n",top,wfact,eta); */
//	printf("Eta=%.4f\n",eta);

      	for(i=0;i<N;i++)
	{	for(j=0;j<N;j++)
		{	z=holo[i][j][0]/top;
		        if(fabs(z)>=eta)
		            holo[i][j][0]=(z>=0.0) ? 1.0 : -1.0;
			else
			    holo[i][j][0]=z;
		}
	}
}

/*****************************************************************************/
/*****************************************************************************/

void holoconstrain3(eta)

double eta;

{	extern fftw_complex holo[N][N];
	double x,z,mean,squares,top,sigma,limit,wfact,twopi;
	extern int iter,count;
	int i,j;
	unsigned char c;
	FILE *fp;

	mean=0.0;squares=0.0;top=0.0;

	for(i=0;i<N;i++)
	{	for(j=0;j<N;j++)
		{     z=holo[i][j][0]*cos(holo[i][j][1]);   /* real part */ 
		      holo[i][j][0]=z;holo[i][j][1]=0.0;
		      z=fabs(z);
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
	wfact=squares/(top*top*N*N);
	mean/=(N*N);squares/=(N*N);
	sigma=sqrt(squares-(mean*mean));
	limit=mean+sigma;
	//	printf("Meanhol=%.3f  sighol=%.3f  ",mean,sigma);
	//printf("tophol=%.3f  Wfact=%f  eta=%.4f\n",top,wfact,eta);

      	for(i=0;i<N;i++)
	{	for(j=0;j<N;j++)
		{	z=holo[i][j][0]/top;
		        if(fabs(z)>=eta)
		            holo[i][j][0]=(z>=0.0) ? 1.0 : -1.0;
			else
			    holo[i][j][0]=z;
		}
	}
}

/*****************************************************************************/
/*****************************************************************************/

void outhol(filename)

char *filename;

{	extern fftw_complex holo[N][N];
	int i,j,d,hist[256];
	unsigned char c;
	FILE *fp;

	fp=fopen(filename,"w");
  	fprintf(fp,"P5\n# CREATOR: iftabin.c : K.Heggarty Dept Optique\n");
	fprintf(fp,"%d %d\n",N,N);
        fprintf(fp,"255\n");

	for(i=0;i<256;i++)
	  hist[i]=0;

	for(i=0;i<N;i++)
	{	for(j=0;j<N;j++)
		{	c=(holo[i][j][0]==1.0) ? 128 : 0;
			fputc(c,fp);
			hist[c]=hist[c]+1;
		}
	}
	fclose(fp);

	for(i=0;i<2;i++)       /* Histogram of phase levels */
	  {       //printf("hist[%d] = %d\n",i*128,hist[i*128]);
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
	//printf("cross=%f\n",cross);
	//	        printf("lambda=%f\n",lambda);

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
