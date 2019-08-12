#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <list>
#include <vector>
#include <utility>
#include <algorithm>
#include <omp.h>

// Global data

typedef float FLOAT;
const FLOAT zero=0.0;
const FLOAT two=2.0;
const FLOAT half=0.5;

const int natom = 8000;//*4;
const FLOAT sigma = 20;	// Particle radius
const FLOAT rsigmasq = 1.0/(sigma*sigma);
const FLOAT L = 1600;//*2;		// Box size
const FLOAT epsilon = 1.0;	// Binding energy
const FLOAT target_temp = 0.4;
const int nprint = 100;		// Print & temp scale every 100 steps
const int nneigh = 20;		// Recompute neighbor list every 20 steps
const int nstep=1000;		// Number of steps to take

const FLOAT r2cut_neigh = 35;
const FLOAT r2cut_force = 30;
const FLOAT excess_vel = 1.6;

FLOAT time_force, time_neigh, time_total;

typedef std::vector<FLOAT> vectorT;

typedef std::vector<std::vector<int>> neighT;

inline FLOAT periodic(FLOAT x, FLOAT L) {
    while (x>L) x-=L;
    while (x<0) x+=L;
    return x;
}

void neighbor_list(const vectorT& X, const vectorT& Y, neighT& neigh) {
    static size_t maxlist = 180;
    double start = omp_get_wtime();
    
#pragma omp parallel default(none) shared(neigh, X, Y, maxlist)
    {
      int mask[natom]; // NOT vector<bool> since that is sloooow
      const int nthread = omp_get_num_threads();
      const int tid = omp_get_thread_num();
      for (int i=tid; i<natom; i+=nthread) {  // Loop should match forces for locality
	std::vector<int>& jlist = neigh[i];
	jlist.clear();
	jlist.reserve(maxlist);
	FLOAT xi = X[i];
	FLOAT yi = Y[i];
	
#pragma omp simd
	for (int j=i+1; j<natom; j++) {
	  FLOAT xj = X[j];
	  FLOAT yj = Y[j];
	  FLOAT dx = (xi-xj);
	  FLOAT dy = (yi-yj);
	  
	  if (dx > (L/2)) dx = dx - L;
	  else if (dx < (-L/2)) dx = dx + L;
	  
	  if (dy > (L/2)) dy = dy - L;
	  else if (dy < (-L/2)) dy = dy + L;
	  
	  FLOAT r2 = (dx*dx+dy*dy)*rsigmasq;
	  mask[j] = (r2 < r2cut_neigh);
	}
	
	for (int j=i+1; j<natom; j++) {
	  if (mask[j]) jlist.push_back(j);
	}
      }
    }

    time_neigh += omp_get_wtime() - start;
    maxlist = 0;
    for (int i=0; i<natom; i++) {
        std::vector<int>& jlist = neigh[i];
	maxlist = std::max(maxlist,jlist.size());
    }
    maxlist *= FLOAT(1.1);
}

void morton_order(vectorT& X, vectorT& Y) {
  struct Fred {
    FLOAT x, y;
    Fred(FLOAT x, FLOAT y) : x(x), y(y) {}
    Fred(){}
    bool operator<(const Fred& other) const {
      static const FLOAT h = L/64;
      int a=x/h, b=y/h;
      int c=other.x/h, d=other.y/h;
      int ac = a^c;
      int bd = b^d;
      if (ac<bd && ac<(ac^bd)) {
	return (b<d);
      }
      else {
	return (a<c);
      }
    }
  };
  std::vector<Fred> s(natom);
  for(int i=0; i<natom; i++) s[i] = Fred(X[i],Y[i]);
  std::sort(s.begin(),s.end());
  for(int i=0; i<natom; i++) {X[i]=s[i].x; Y[i]=s[i].y;}
}


void forces(const neighT& neigh, const vectorT& x, const vectorT& y, vectorT& total_fx, vectorT& total_fy, FLOAT& total_virial, FLOAT& total_pe) {
  double start = omp_get_wtime();
  
  // Zero the accumulators
  total_virial = total_pe = zero;
  for (int i=0; i<natom; i++) {
    total_fx[i] = zero;
    total_fy[i] = zero;
  }
  
  // V(ri-rj) = epsilon*((sigma/r)^12 - 2*(sigma/r)^6)
  // dV/dxi = -12*epsilon*((sigma/r)^14 - (sigma/r)^8)*(xi-xj)/sigma**2
  // F[i][x] = -dV/dxi
  const FLOAT fac = epsilon*12.0/(sigma*sigma);
  
#pragma omp parallel default(none) shared(total_fx, total_fy, neigh, x, y, total_virial, total_pe)
  {
    // Thread-local accumulators
    FLOAT virial=zero, pe=zero;
    FLOAT fx[natom], fy[natom];
    for (int i=0; i<natom; i++) {
      fx[i] = zero;
      fy[i] = zero;
    }

    // Thread-local vectors to gather coordinates and forces
    FLOAT Xj[natom], Yj[natom], Fxj[natom], Fyj[natom];

    const int nthread = omp_get_num_threads();
    const int tid = omp_get_thread_num();
    for (int i=tid; i<natom; i+=nthread) {  // Loop should match NL for locality
      const std::vector<int>& jlist = neigh[i];
      const int nij = jlist.size();
      
      for (int ij=0; ij<nij; ij++) {
	const int j = jlist[ij];
	Xj[ij] = x[j];
	Yj[ij] = y[j];
      }
      for (int ij=0; ij<nij; ij++) {
	Fxj[ij] = zero;
	Fyj[ij] = zero;
      }
      const FLOAT xi = x[i];
      const FLOAT yi = y[i];
      FLOAT fxi = zero;
      FLOAT fyi = zero;
#pragma omp simd reduction(+:fxi,fyi,pe,virial)
      for (int ij=0; ij<nij; ij++) {
	const FLOAT xj = Xj[ij];
	const FLOAT yj = Yj[ij];
	
	FLOAT dx = (xi-xj);
	FLOAT dy = (yi-yj);
	
	if (dx > (L/2)) dx = dx - L;
	else if (dx < (-L/2)) dx = dx + L;
	
	if (dy > (L/2)) dy = dy - L;
	else if (dy < (-L/2)) dy = dy + L;
	
	FLOAT r2 = (dx*dx + dy*dy)*rsigmasq;
	
	if (r2 < r2cut_force) {
	  // FLOAT r6 = r2*r2*r2;
	  // FLOAT r12 = r6*r6;
	  // FLOAT vij = epsilon*(1.0/r12 - two/r6);
	  // FLOAT df = fac*(1.0/(r12*r2) - 1.0/(r6*r2));
	  
	  FLOAT rr2 = 1.0/r2;
	  FLOAT rr6 = rr2*rr2*rr2;
	  FLOAT rr12 = rr6*rr6;
	  FLOAT vij = epsilon*(rr12 - two*rr6);
	  FLOAT df = fac*(rr12 - rr6)*rr2;
	  
	  FLOAT dfx = df*dx;
	  FLOAT dfy = df*dy;
	  
	  fxi += dfx;
	  fyi += dfy;
	  Fxj[ij] -= dfx;
	  Fyj[ij] -= dfy;
	  
	  pe += vij;
	  virial += dfx*dx + dfy*dy;
	}
      }
      fx[i] += fxi;
      fy[i] += fyi;
      for (int ij=0; ij<nij; ij++) {
	const int j = jlist[ij];
	fx[j] += Fxj[ij];
	fy[j] += Fyj[ij];
      }
    }
#pragma omp critical
    {
      total_virial += virial;
      total_pe += pe;
      for (int i=0; i<natom/4; i++) {
	total_fx[i] += fx[i];
	total_fy[i] += fy[i];
      }
    }
#pragma omp critical 
    for (int i=natom/4; i<natom/2; i++) {
      total_fx[i] += fx[i];
      total_fy[i] += fy[i];
    }
#pragma omp critical 
    for (int i=natom/2; i<3*natom/4; i++) {
      total_fx[i] += fx[i];
      total_fy[i] += fy[i];
    }
#pragma omp critical 
    for (int i=3*natom/4; i<natom; i++) {
      total_fx[i] += fx[i];
      total_fy[i] += fy[i];
    }
  }
  time_force += omp_get_wtime() - start;
}

inline FLOAT restrict(FLOAT a, FLOAT b) {
    if (a > b) return b;
    else if (a < -b) return -b;
    else return a;
}

void optimize(vectorT& X, vectorT& Y, vectorT& Fx, vectorT& Fy, neighT& neigh) {
    FLOAT dt = 0.1;
    FLOAT prev = 1e30;
    for (int step=0; step<600; step++) {
      if ((step%(3*nneigh)) == 0 || step<10) neighbor_list(X, Y, neigh);
        FLOAT virial,pe;
        forces(neigh,X,Y,Fx,Fy,virial,pe);
        for (int i=0; i<natom; i++) {
	    FLOAT x = X[i];
            FLOAT y = Y[i];
            FLOAT fx= restrict(Fx[i], two);
            FLOAT fy= restrict(Fy[i], two);
	    X[i] = periodic(x+dt*fx,L);
	    Y[i] = periodic(y+dt*fy,L);
        }

        if ((step%50)==0) 
	  std::cout << "optim: " <<  pe << " " <<  dt << std::endl;
        
        if (std::abs(pe-prev) < std::abs(prev*FLOAT(0.00001))) break;
        prev = pe;
    }
}

FLOAT drand() {
    static const unsigned int a = 1664525;
    static const unsigned int c = 1013904223;
    static unsigned int x = 23111;
    static const FLOAT fac = 2.3283064365386963e-10;

    x = a*x + c;
    
    return fac*x;
}


void md() {
    const FLOAT dt=0.03;
    std::cout << "Time step " << dt << std::endl;

    // initialize random coords and velocities
    vectorT X(natom), Y(natom), Fx(natom), Fy(natom), Vx(natom), Vy(natom);

    FLOAT vxmean=zero, vymean= zero;

    // int n2 = sqrt(FLOAT(natom))+1;
    // FLOAT h = L/n2;
    // int n = 0;
    // for (int i=0; i<n2; i++) {
    //   for (int j=0; j<n2; j++) {
    // 	X[n] = (i+half)*h;
    // 	Y[n] = (j+half)*h;
    // 	n++;
    // 	if (n == natom) break;
    //   }
    // }
    // for (int i=0; i<natom; i++) {
    //     FLOAT vx = (drand()-half)*std::sqrt(two*target_temp)*two*excess_vel;
    //     FLOAT vy = (drand()-half)*std::sqrt(two*target_temp)*two*excess_vel;
    //     vxmean += vx;
    //     vymean += vy;
    //     Vx[i] = vx;
    // 	Vy[i] = vy;
    // }

    FLOAT box = std::min(std::sqrt(FLOAT(natom))*sigma*FLOAT(1.25),L);
    for (int i=0; i<natom; i++) {
        FLOAT xi, yi;
        for (int attempt=0; attempt<200; attempt++) {
            xi = box*(drand()-half) + L*half;
            yi = box*(drand()-half) + L*half;
            FLOAT r2min = FLOAT(1000000.0);
            for (int j=0; j<i; j++) {
                FLOAT xj = X[j];
                FLOAT yj = X[j];
                FLOAT dx = (xi-xj);
                FLOAT dy = (yi-yj);
                r2min = std::min(r2min,dx*dx+dy*dy);
            }
            if (r2min > half*sigma*sigma) break;
        }
        //std::cout << xi << " " << yi << std::endl;
    	X[i] = xi;
    	Y[i] = yi;
        FLOAT vx = (drand()-half)*std::sqrt(two*target_temp)*two*excess_vel;
        FLOAT vy = (drand()-half)*std::sqrt(two*target_temp)*two*excess_vel;
        vxmean += vx;
        vymean += vy;
        Vx[i] = vx;
    	Vy[i] = vy;
    }
    
    vxmean /= natom;
    vymean /= natom;
    
    for (int i=0; i<natom; i++) {
        Vx[i] -= vxmean;
        Vy[i] -= vymean;
    }

    morton_order(X,Y);

    neighT neigh(natom);
    neighbor_list(X,Y,neigh);

    optimize(X,Y,Fx,Fy,neigh);
    morton_order(X,Y);

    // make the initial forces
    FLOAT virial = zero;
    FLOAT temp = zero;

    FLOAT potential_energy;
    neighbor_list(X,Y,neigh);
    forces(neigh,X,Y,Fx,Fy,virial,potential_energy);
    
    int step_stats = 0;
    for (int step=1; step<=nstep; step++) {
        // update the velocities to time t+dt/2 and the positions to time t+dt
        for (int i=0; i<natom; i++) {
   	    FLOAT vx = Vx[i], vy = Vy[i];
            FLOAT fx = Fx[i], fy = Fy[i];
	    FLOAT x = X[i], y = Y[i];
            vx += fx*dt*half;
            vy += fy*dt*half;
            x += vx*dt;
            y += vy*dt;
	    Vx[i] = vx;
	    Vy[i] = vy;
	    X[i] = periodic(x,L);
	    Y[i] = periodic(y,L);
        }
        // make the forces at time t+dt
        if ((step%nneigh) == 0) {
	  neighbor_list(X, Y, neigh);
        }

        FLOAT virial_step;
        forces(neigh,X,Y,Fx,Fy,virial_step,potential_energy);
        virial += virial_step;

        // finish update of v to time t+dt
        FLOAT kinetic_energy = zero;
        for (int i=0; i<natom; i++) {
   	    FLOAT vx = Vx[i], vy = Vy[i];
            FLOAT fx = Fx[i], fy = Fy[i];
            vx += fx*dt*half;
            vy += fy*dt*half;
	    Vx[i] = vx;
	    Vy[i] = vy;
            kinetic_energy += half*(vx*vx+vy*vy);
        }

        temp += kinetic_energy/(natom - 1.0);
        step_stats += 1;

        if ((step%nprint) == 0) {

            if (step == nprint) {
                printf("\n");
                printf("    time         ke            pe             e            T          P\n");
                printf("  -------    -----------   ------------  ------------    ------    ------\n");
            }

            temp = temp/step_stats;
            virial = half*virial/step_stats;
            FLOAT pressure = (natom*temp + virial)/(L*L);
            FLOAT energy = kinetic_energy + potential_energy;

            FLOAT vscale = std::sqrt(target_temp/temp);
            if (step>=(nstep/3)) vscale=1.0;
            const char scaling[2]={' ','*'};

            printf("%9.2f   %12.5f   %12.5f  %12.5f %8.3f %12.8f %c\n",
                   step*dt, kinetic_energy, potential_energy, energy, temp, 
                   pressure, scaling[vscale!=1.0]);

            for (int i=0; i<natom; i++) {
                Vx[i] *= vscale;
                Vy[i] *= vscale;
            }

            temp = virial = zero;
            step_stats = 0;
        }
    }
    
}

int main() {
    time_force = time_neigh = time_total = zero;
    double start = omp_get_wtime();
    md();
    time_total += omp_get_wtime() - start;

    int nthread;
#pragma omp parallel shared(nthread)
    nthread = omp_get_num_threads();

    printf("times:  force=%5.2fs  neigh=%5.2fs  total=%5.2fs  nthread=%d\n", 
	   time_force, time_neigh, time_total, nthread);

    return 0;
}
