#include <mpi.h>
#include <omp.h>

#include <iostream>
#include <algorithm>
#include <vector>

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cassert>

#ifdef ENABLE_XDEBUG
#include <sys/types.h>
#include <unistd.h>

// Another place to catch the debugger
extern "C" void xterm_debug_breakpoint() {
    std::cout << "xterm_debug_breakpoint" << std::endl;
}

// Invoke with path to executable and X-window display name to start
// separate xterm for each process with debugger attached.
//

void xterm_debug(const char* path, const char* display) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);    

    pid_t child;
    const char *argv[20], *xterm = "/usr/bin/xterm"; // Might need changing on your system
    char title[256], pid[256], geometry[256];
    int ix=(rank/3)%3;
    int iy=rank%3;
    sprintf(title, "Debugging process %d ", rank);
    sprintf(pid, "%d", getpid());
    sprintf(geometry,"%dx%d+%d+%d",80,24,ix*500,iy*280);

    if (path == 0) path = "test1";
    if (display == 0) display = getenv("DISPLAY");
    if (display == 0) return ;

    argv[0] = xterm;
    argv[1] = "-T";
    argv[2] = title;
    argv[3] = "-display";
    argv[4] = display;
    argv[5] = "-fn";
    argv[6] = "6x10";
    argv[7] = "-geometry";
    argv[8] = geometry;
    argv[9] = "-e";
    argv[10] = "gdb";
    argv[11] = "-q";
    argv[12] = path;
    argv[13] = pid;
    argv[14] = 0;
    if (rank == 0) {
        int i;
        printf("\n Starting xterms with debugger using command\n\n    ");
        for (i = 0; argv[i]; ++i) printf("%s ", argv[i]);
        printf("\n\n");
        fflush(stdout);
    }
    
    child = fork();
    
    if (child < 0) {
        printf("debug: fork failed?\n\n");
    }
    else if (child > 0) {
        sleep(20);			/* Release cpu while debugger starts*/
        xterm_debug_breakpoint();
    }
    else {
        execv(xterm, (char*const*) argv);
        perror("");
        printf("util_debug: execv of xterm with debugger failed\n\n");
    }
}
#endif // ENABLE_XDEBUG

// Some globals for convenience/laziness/readability

static const double LO = -3.1415926535;       // Phsyical dimensions of the domain are [LO,HI]^2
static const double HI =  3.1415926535;

int P, ncol_P, nrow_P;   // P = No. of processes = ncol_P * nrow_P
int my_col_P, my_row_P;  // Co-ords of this proc. in grid of processes
int rank;                // MPI rank of this process 

double * __restrict__ buff1;                 // Buffer used for exchanging rows
double * __restrict__ buff2;                 // Buffer used for exchanging rows

int north, south, east, west;  // Rank of process in that direction on process grid or -1 if no-one there

void Error(const char* msg, int code) {
    std::cerr<<msg<< std::endl;
    MPI_Abort(MPI_COMM_WORLD,code);
}

/// Simple wrapper class to make 2D mesh easier to use
class Grid {
    int nrow, ncol;             // Number of rows/columns
    int row_lo, col_lo;         // [0][0] in this local grid is [row_lo][col_low] in the global grid
    double h;                   // Size of box in the grid
    std::vector<double> data;   // The actual data [i][j] --> data[i*ncol+j]
public:
    Grid(int nrow, int ncol, int row_lo, int col_lo, double h) : nrow(nrow), ncol(ncol), row_lo(row_lo), col_lo(col_lo), h(h), data(ncol*nrow,-9.999) {}
    Grid() : nrow(0), ncol(0), row_lo(0), col_lo(0), h(0), data() {}
    inline double* operator[](int i){return &data[i*ncol];}
    inline const double* operator[](int i)const{return &data[i*ncol];}
    inline double row_coord(int i) const {return h*(i+row_lo) + LO;}
    inline double col_coord(int i) const {return h*(i+col_lo) + LO;}
    inline int ncols() const {return ncol;}
    inline int nrows() const {return nrow;}
    inline int row_low() const {return row_lo;}
    inline int col_low() const {return col_lo;}
    inline double scale() const {return h;}
};

double time_exchange = 0;            /* Timing information */
double time_global = 0;
double time_interpolate = 0;
double time_operate = 0;
double time_total = 0;

// Model potential that defines solution and boundary conditions.
// This particular choice makes the discretization error quite
// apparent.  Needs to be a harmonic function (i.e., satisfy Laplace's
// equation)
double Solution(double x, double y)
{
    return 0.5 * (cos(x)*exp(-y) + cos(y)*exp(-x) +
                  0.1*(sin(2.*x)*exp(-2.*y) + sin(2.*y)*exp(-2.*x)) +
                  0.1*(cos(3.*x)*exp(-3.*y) + cos(3.*y)*exp(-3.*x))) * 0.001; // 0.001 just to make printing of grid values pretty
    //return x*y; (x + y + x*y + x*x - y*y);
}

// Compute mean absolute error at grid points relative to the
// analytic solution ... error is due to either lack of convergence
// or discretization error.
double GridError(const Grid& grid, int ngrid)
{
    const int nrows = grid.nrows()-1, ncols = grid.ncols()-1;
    
    double error = 0.0;
#pragma omp parallel for reduction(+:error)
    for (int i=1; i<nrows; i++)
#pragma omp simd reduction(+:error)
        for (int j=1; j<ncols; j++)
            error += fabs(grid[i][j] - Solution(grid.row_coord(i), grid.col_coord(j)));
    
    double start = MPI_Wtime();
    double total_error;
    MPI_Allreduce(&error, &total_error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    time_global += MPI_Wtime() - start;
    
    return total_error / (double(ngrid-1)*(ngrid-1));
}

//  Fill in b.c.s on a grid
void BoundaryConditions(Grid& grid)
{
    const int nrows = grid.nrows()-1, ncols = grid.ncols()-1;
    for (int i=0; i<=nrows; i++) {
        if (my_col_P == 0)
            grid[i][0]  = Solution(grid.row_coord(i), LO);
        if (my_col_P == (ncol_P-1))
            grid[i][ncols] = Solution(grid.row_coord(i), HI);
    }
    
    for (int j=0; j<=ncols; j++) {
        if (my_row_P == 0)
            grid[0][j]  = Solution(LO, grid.col_coord(j));
        if (my_row_P == (nrow_P-1))
            grid[nrows][j] = Solution(HI, grid.col_coord(j));
    }
}

// Fill in boundary values and zero initial guess for interior.
void Initialize(Grid& grid)
{
    const int nrows = grid.nrows()-1, ncols = grid.ncols()-1;

    BoundaryConditions(grid);

    for (int i=1; i<nrows; i++)
        for (int j=1; j<ncols; j++)
            grid[i][j] = 0.0; //(i+row_low) + (j+grid.col_low())*0.001; //for debug
}

// Exchange data with neighboring ndoes. In Operate only need to
// exchange red and black elements separately but do NOT do this at
// the moment ... thus are sending twice as much data as
// necessary. Interpolate needs to exchange the full boundary
// information.
void Exchange(Grid& grid)
{
    const int nrows = grid.nrows()-1, ncols = grid.ncols()-1;
    const int type1=1, type2=2, type3=3, type4=4, type5=5, type6=6, type7=7, type8=8;
    const int bnrows = (nrows+1)*sizeof(double);
    const int bncols = (ncols+1)*sizeof(double);
    double start = MPI_Wtime();
    
#define GATHER(k)  for (int i=0; i<=nrows; i++) buff1[i] = grid[i][k]
#define SCATTER(k) for (int i=0; i<=nrows; i++) grid[i][k] = buff2[i]
    
    MPI_Status status;
    if (my_row_P%2) {
        if (north >= 0) {        
            MPI_Sendrecv(grid[1], bncols, MPI_BYTE, north, type1,
                         grid[0], bncols, MPI_BYTE, north, type2, MPI_COMM_WORLD, &status);
        }
        if (south >= 0) {
            MPI_Sendrecv(grid[nrows-1], bncols, MPI_BYTE, south, type3,
                         grid[nrows  ], bncols, MPI_BYTE, south, type4, MPI_COMM_WORLD, &status);
        }
    }
    else {
        if (south >= 0) {
            MPI_Sendrecv(grid[nrows-1], bncols, MPI_BYTE, south, type2,
                         grid[nrows  ], bncols, MPI_BYTE, south, type1, MPI_COMM_WORLD, &status);
        }
        if (north >= 0) {
            MPI_Sendrecv(grid[1], bncols, MPI_BYTE, north, type4,
                         grid[0], bncols, MPI_BYTE, north, type3, MPI_COMM_WORLD, &status);
        }
    }
    
    if (my_col_P%2) {
        if (west >= 0) {
            GATHER(1);
            MPI_Sendrecv(buff1, bnrows, MPI_BYTE, west, type5,
                         buff2, bnrows, MPI_BYTE, west, type6, MPI_COMM_WORLD, &status);
            SCATTER(0);
        }
        if (east >= 0) {
            GATHER(ncols-1);
            MPI_Sendrecv(buff1, bnrows, MPI_BYTE, east, type7,
                         buff2, bnrows, MPI_BYTE, east, type8, MPI_COMM_WORLD, &status);
            SCATTER(ncols);
        }
    }
    else {
        if (east >= 0) {
            GATHER(ncols-1);
            MPI_Sendrecv(buff1, bnrows, MPI_BYTE, east, type6,
                         buff2, bnrows, MPI_BYTE, east, type5, MPI_COMM_WORLD, &status);
            SCATTER(ncols);
        }
        if (west >= 0) {
            GATHER(1);
            MPI_Sendrecv(buff1, bnrows, MPI_BYTE, west, type8,
                         buff2, bnrows, MPI_BYTE, west, type7, MPI_COMM_WORLD, &status);
            SCATTER(0);
            
        }
    }
    time_exchange += MPI_Wtime() - start;

#undef GATHER
#undef SCATTER
 }
 
void PrintGrid(const Grid& grid)
{
     const int nrows = grid.nrows()-1, ncols = grid.ncols()-1;
     
     printf("         ");
     for (int j=0; j<=ncols; j++) printf("%4d   ", j+grid.col_low());
     printf("\n");
     for(int i=0; i<=nrows; i++) {
         printf("%6d  ", i+grid.row_low());
         for (int j=0; j<=ncols; j++) {
             printf("%6.3f ",grid[i][j]);
         }
         printf("\n");
         fflush(stdout);
     }
}


// Update grid in place according to the simple rule
//
// for 0<i<nrows, 0<j<ncols.
//    new[i][j] = 0.25 * 
//      (old[i-1][j] + old[i+1][j] + old[i][j-1] + old[i][j+1]);
//
// To increase parallelism and also improve convergence rate for large
// grids use red/black checkerboard -> update red then update black
// using new reds (gauss-seidel red-black w-relaxation).
//
// Return the mean abs. error (value of the laplacian) on the old grid.
double Operate(Grid& grid, int ngrid, int do_sums, double omega)
{
    const int nrows = grid.nrows()-1, ncols = grid.ncols()-1, row_low = grid.row_low(), col_low = grid.col_low();
    double start = MPI_Wtime();
    double change = 0.0;
    
    for (int color=0; color<=1; color++) { //0=black, 1=red
        Exchange(grid);

#pragma omp parallel for reduction(+:change) 
        for (int i=1; i<nrows; i++) {
            int jlo = 1+((i-color+col_low+row_low)%2);
            double* __restrict__ gg = grid[i];
            const double* ggm = grid[i-1];
            const double* ggp = grid[i+1];
            
#pragma omp simd reduction(+:change)            
            for (int j=jlo; j<ncols; j+=2) {
                double newval = 0.25 * (ggp[j] + ggm[j] + gg[j-1] + gg[j+1]);
                double diff = newval - gg[j];
                newval = newval + omega*diff;
                change += fabs(diff);
                gg[j] = newval;
            }
        }
    }
    
    if (do_sums) {
        double start = MPI_Wtime();
        double total_change;
        MPI_Allreduce(&change, &total_change, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        time_global += MPI_Wtime() - start;

        change = 4.0 * total_change / (grid.scale()*grid.scale()*double(ngrid-2)*(ngrid-2));
    }
    else {
        change = 999999.0;
    }
    
    time_operate += MPI_Wtime() - start;

    return change;
}
 
//  Bilinear interpolation to higher resolution new grid
void Interpolate(const Grid& old_grid, Grid& new_grid) 
{
    const int old_nrows = old_grid.nrows()-1, old_ncols = old_grid.ncols()-1;
    const int new_nrows = new_grid.nrows()-1, new_ncols = new_grid.ncols()-1;
    const double r_old_h = 1.0/old_grid.scale();
    const double r_new_h = 1.0/new_grid.scale();
    
    double start = MPI_Wtime();
    BoundaryConditions(new_grid);
    
    for (int i=1; i<new_nrows; i++) {
        int iold = i/2;
        double xnew = new_grid.row_coord(i);
        double xold_left = old_grid.row_coord(iold);
        double x = (xnew-xold_left)*r_old_h;
        for (int j=1; j<new_ncols; j++) {
            int jold = j/2;
            double ynew = new_grid.col_coord(j);
            double yold_left = old_grid.col_coord(jold);
            double y = (ynew-yold_left)*r_old_h;
            double f00 = old_grid[iold  ][jold  ];
            double f10 = old_grid[iold+1][jold  ];
            double f01 = old_grid[iold  ][jold+1];
            double f11 = old_grid[iold+1][jold+1];

            new_grid[i][j] = (f00*(1-x) + f10*x)*(1-y) + (f01*(1-x) + f11*x)*y;
        }
    }
    
    Exchange(new_grid);
    
    time_interpolate += MPI_Wtime() - start;
}

// Apply iterative procedure to solve current grid.
// Parallel version only does global sums every nsums iterations.
void Solve(Grid& grid, int ngrid, int niter, int nprint, double thresh)
{
    const int nsums = std::min(10, nprint); // Need sums whenever we print
    double omegatarget = 1.0 - 9.86/ngrid;
    double omega = 0.0;
    
    if (nprint%nsums)
        nprint= nprint + nsums - (nprint%nsums); // Make nprint a multiple of nsums
                                                    
    for (int iter=0; iter<niter; iter++) {
        bool do_sums = !(iter%nsums); // For efficiency only do global sums every 10 iters

        double change = Operate(grid, ngrid, do_sums, omega);  // Actually do the work

        if ((iter+1)%30 == 0) omega = (omega + omegatarget)*0.5;
        
        if ((rank==0) && ((iter%nprint == 0) || (change < thresh))) {
            (void) printf("ngrid=%4d iter=%4d change=%.2e\n", ngrid, iter+1, change);
            (void) fflush(stdout);
        }
        
        // Are we converged?
        if (do_sums && (change < thresh)) {
            if (rank == 0) {
                (void) printf("Converged!\n");
                (void) fflush(stdout);
            }
            break;
        }
    }
    
    Exchange(grid); // inal exchange to get edges correct
    
    double error = GridError(grid, ngrid);
    if (rank == 0) {
        (void) printf("Mean abs. error to exact soln. = %.2e, ngrid=%d\n\n",
                      error, ngrid);
        (void) fflush(stdout);
    }
}

void ParseArguments(int argc, char** argv, int& ngrid, int& niter, int& nprint, int& nlevel, double& thresh, bool& debug, bool& rio, bool& pgrid)
 {
     argv++; argc--; // pop executable name
     while (argc) {
         if (strcmp(*argv, "-niter") == 0) {niter = atoi(*++argv); argc--;}
         else if (strcmp(*argv, "-ngrid") == 0) {ngrid = atoi(*++argv); argc--;}
         else if (strcmp(*argv, "-nprint") == 0) {nprint = atoi(*++argv); argc--;}
         else if (strcmp(*argv, "-nlevel") == 0) {nlevel = atoi(*++argv); argc--;}
         else if (strcmp(*argv, "-thresh") == 0) {(void) sscanf(*++argv, "%lf", &thresh); argc--;}
         else if (strcmp(*argv, "-debug") == 0) {debug = true;}
         else if (strcmp(*argv, "-rio") == 0) {rio = true;}
         else if (strcmp(*argv, "-pgrid") == 0) {pgrid = true;}
         else if (strcmp(*argv, "-help") == 0) {
             (void) fprintf(stderr,"gridtest [-ngrid #] [-nprint #] [-niter #]\n");
             (void) fprintf(stderr,"         [-thresh #] [-nlevel #] [-help] [-debug]\n");
             Error("failed parsing arguments",1);
         }
         else {
             fprintf(stderr, "unknown argument %s\n", *argv);
             Error("failed parsing arguments",1);
         }             
         argv++; argc--;
     }
 }

// Factor N into two integers that are as close together as possible
 void Factor(int N, int& n, int& m)
{
    m = int(sqrt((double) N));
    while (N%m) m--;
    n = N/m;
}

void Partition(int ngrid, int npart, int ipart, int& n, int& lo)
{
    int chunk = (ngrid-2)/npart; // Main program made sure this was an exact multiple
    n = chunk+2;
    lo = ipart*chunk;
}

// Redirect stdout and stderr to file "log.<rank>"
void redirectio(bool split=true) {
    char filename[256];
    std::sprintf(filename,"log.%5.5d",rank);
    char errfilename[256];
    std::sprintf(errfilename,"%s.%5.5d", (split ? "err" : "log"), rank);
    if (!freopen(   filename, "w", stdout)) Error("reopening stdout failed", 1);
    if (!freopen(errfilename, "w", stderr)) Error("reopening stderr failed", 2);
    std::cout.sync_with_stdio(true);
    std::cerr.sync_with_stdio(true);
}

// Grid test code. Solve Laplaces eqn. on a square grid subject
// to b.c.s on the boundary.  Use 5 point discretization of the
// operator and a heirarchy of grids with red/black gauss seidel
// w-relaxation (omega=0.94 as 1.0 diverges).

// command arguments:

// -ngrid  # = dimension of largest grid (default = 1922)
// -niter  # = max. no. of interations   (default = 4000)
// -nprint # = print change every nprint iterations (default = 100)
// -nlevel # = no. of grid levels        (default = 4)
// -thresh # = convergence criterion on the change between iterations (default = 0.00001)
// -rio      = if rank>0 redirect IO to per-process log files (useful if printing grid)
// -debug    = attach debugger to each process inside an xterm if built with ENABLE_XDEBUG
// -pgrid    = print the grid before and after solution at each resolution (forces ngrid<=64)
// -help     = print usage and exit with error
int main(int argc, char** argv)
{
    // Defaults for command line options --- see above for documentation
    int maxgrid=((3*5*8)<<4) + 2; // So will get same grid and hence consistent numerics and timings on nproc=1,2,3,4,5,6,8,10,12,15,20,24,30,40,60,120
    int niter=4000;
    int nprint=100;
    int nlevel=4;
    double thresh=0.00001;
    bool rio=false;
    bool debug=false;
    bool pgrid=false;
    
    int threadlevel;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_SINGLE,&threadlevel);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    ParseArguments(argc, argv, maxgrid, niter, nprint, nlevel, thresh, debug, rio, pgrid);
    if (rio && rank) redirectio();
    if (pgrid) maxgrid = std::min(maxgrid,64); // Otherwise you will drown in output
    
    double start = MPI_Wtime();
    
    Factor(P, nrow_P, ncol_P);    /* Arrange processes into a grid   */
    my_row_P = rank/ncol_P;
    my_col_P = rank - my_row_P*ncol_P;
    if (rio || rank==0) printf("%2d: Process grid: %dx%d (%d,%d)\n", rank, nrow_P, ncol_P, my_row_P, my_col_P);
    
    // Who are my neighbors on the process grid?
    north = (my_row_P > 0)          ? rank-ncol_P : -1;
    south = (my_row_P < (nrow_P-1)) ? rank+ncol_P : -1;
    east  = (my_col_P < (ncol_P-1)) ? rank+1      : -1;
    west  = (my_col_P > 0)          ? rank-1      : -1;
    
    if (rio || rank==0) printf("%2d: north=%d south=%d east=%d west=%d\n", rank, north, south, east, west);
    
    // Make the smallest number of interior grid points a multiple of
    // size of the processor grid in both dimensions to make the code
    // logic easier and for better load balance
    int ngrid1 = std::max(1,(maxgrid-2)>>(nlevel-1));
    while (!((ngrid1%nrow_P)==0 && (ngrid1%ncol_P)==0)) ngrid1++;
    maxgrid = (ngrid1<<(nlevel-1)) + 2;     // Actual size of final grid including boundary points
    
    buff1 = new double[maxgrid];
    buff2 = new double[maxgrid];
    
#ifdef ENABLE_XDEBUG
    if (debug) xterm_debug(argv[0], ":0");
#endif

    // Loop from coarse to fine grids
    Grid grid;
    for (int level=0; level<nlevel; level++) {
        int ngrid = (ngrid1<<level) + 2;        // Grid dimension including boundary values
        double scale = (HI - LO) / (ngrid-1); 
        
        // Partition rows and columns between processors.  The square
        // mesh has (ngrid-2)*(ngrid-2) interior points.  Divide these
        // up equitably between all the processors.
        int nrows, ncols, row_low, col_low;
        Partition(ngrid, nrow_P, my_row_P, nrows, row_low);
        Partition(ngrid, ncol_P, my_col_P, ncols, col_low);
        if (rio || rank==0) printf("%2d: Grid partition: %dx%d (%d,%d)\n", rank, nrows, ncols, row_low, col_low);

        Grid new_grid(nrows, ncols, row_low, col_low, scale);
        if (level == 0) {
            Initialize(new_grid);
        }
        else {
            Interpolate(grid, new_grid);
        }
        grid = new_grid;

        if (pgrid) {
            printf("before solution\n");
            Exchange(grid);
            PrintGrid(grid);
        }
            
        Solve(grid, ngrid, niter, nprint, thresh);

        if (pgrid) {
            printf("after solution\n");
            Exchange(grid);
            PrintGrid(grid);
        }
        fflush(stdout);
    }
    
    time_total = MPI_Wtime() - start;
    
    // Note that the exchange time is double counted inside operate and interpolate
    if (rank == 0) {
        (void) printf("\n #proc  #thread    Operate    Exchange   Global-sum   Interpolate    Total ");
        (void) printf("\n -----  -------    -------    --------   ----------   -----------   -------");
        (void) printf("\n %4d    %4d      %6.2f     %6.2f      %6.2f       %6.2f      %7.2f\n",
                      P, omp_get_max_threads(),
                      time_operate, time_exchange, time_global, time_interpolate, time_total);
    }

    delete [] buff1;
    delete [] buff2;
    
    MPI_Finalize();
    return 0;
}

