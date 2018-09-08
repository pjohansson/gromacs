/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2012,2013,2014,2015,2017,2018, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */
/*
 * Note: this file was generated by the GROMACS avx_256_double kernel generator.
 */
#include "gmxpre.h"

#include "config.h"

#include <math.h>

#include "../nb_kernel.h"
#include "gromacs/gmxlib/nrnb.h"

#include "kernelutil_x86_avx_256_double.h"

/*
 * Gromacs nonbonded kernel:   nb_kernel_ElecRF_VdwNone_GeomP1P1_VF_avx_256_double
 * Electrostatics interaction: ReactionField
 * VdW interaction:            None
 * Geometry:                   Particle-Particle
 * Calculate force/pot:        PotentialAndForce
 */
void
nb_kernel_ElecRF_VdwNone_GeomP1P1_VF_avx_256_double
                    (t_nblist                    * gmx_restrict       nlist,
                     rvec                        * gmx_restrict          xx,
                     rvec                        * gmx_restrict          ff,
                     struct t_forcerec           * gmx_restrict          fr,
                     t_mdatoms                   * gmx_restrict     mdatoms,
                     nb_kernel_data_t gmx_unused * gmx_restrict kernel_data,
                     t_nrnb                      * gmx_restrict        nrnb)
{
    /* Suffixes 0,1,2,3 refer to particle indices for waters in the inner or outer loop, or 
     * just 0 for non-waters.
     * Suffixes A,B,C,D refer to j loop unrolling done with AVX, e.g. for the four different
     * jnr indices corresponding to data put in the four positions in the SIMD register.
     */
    int              i_shift_offset,i_coord_offset,outeriter,inneriter;
    int              j_index_start,j_index_end,jidx,nri,inr,ggid,iidx;
    int              jnrA,jnrB,jnrC,jnrD;
    int              jnrlistA,jnrlistB,jnrlistC,jnrlistD;
    int              jnrlistE,jnrlistF,jnrlistG,jnrlistH;
    int              j_coord_offsetA,j_coord_offsetB,j_coord_offsetC,j_coord_offsetD;
    int              *iinr,*jindex,*jjnr,*shiftidx,*gid;
    real             rcutoff_scalar;
    real             *shiftvec,*fshift,*x,*f;
    real             *fjptrA,*fjptrB,*fjptrC,*fjptrD;
    real             scratch[4*DIM];
    __m256d          tx,ty,tz,fscal,rcutoff,rcutoff2,jidxall;
    real *           vdwioffsetptr0;
    __m256d          ix0,iy0,iz0,fix0,fiy0,fiz0,iq0,isai0;
    int              vdwjidx0A,vdwjidx0B,vdwjidx0C,vdwjidx0D;
    __m256d          jx0,jy0,jz0,fjx0,fjy0,fjz0,jq0,isaj0;
    __m256d          dx00,dy00,dz00,rsq00,rinv00,rinvsq00,r00,qq00,c6_00,c12_00;
    __m256d          velec,felec,velecsum,facel,crf,krf,krf2;
    real             *charge;
    __m256d          dummy_mask,cutoff_mask;
    __m128           tmpmask0,tmpmask1;
    __m256d          signbit = _mm256_castsi256_pd( _mm256_set1_epi32(0x80000000) );
    __m256d          one     = _mm256_set1_pd(1.0);
    __m256d          two     = _mm256_set1_pd(2.0);
    x                = xx[0];
    f                = ff[0];

    nri              = nlist->nri;
    iinr             = nlist->iinr;
    jindex           = nlist->jindex;
    jjnr             = nlist->jjnr;
    shiftidx         = nlist->shift;
    gid              = nlist->gid;
    shiftvec         = fr->shift_vec[0];
    fshift           = fr->fshift[0];
    facel            = _mm256_set1_pd(fr->ic->epsfac);
    charge           = mdatoms->chargeA;
    krf              = _mm256_set1_pd(fr->ic->k_rf);
    krf2             = _mm256_set1_pd(fr->ic->k_rf*2.0);
    crf              = _mm256_set1_pd(fr->ic->c_rf);

    /* Avoid stupid compiler warnings */
    jnrA = jnrB = jnrC = jnrD = 0;
    j_coord_offsetA = 0;
    j_coord_offsetB = 0;
    j_coord_offsetC = 0;
    j_coord_offsetD = 0;

    outeriter        = 0;
    inneriter        = 0;

    for(iidx=0;iidx<4*DIM;iidx++)
    {
        scratch[iidx] = 0.0;
    }

    /* Start outer loop over neighborlists */
    for(iidx=0; iidx<nri; iidx++)
    {
        /* Load shift vector for this list */
        i_shift_offset   = DIM*shiftidx[iidx];

        /* Load limits for loop over neighbors */
        j_index_start    = jindex[iidx];
        j_index_end      = jindex[iidx+1];

        /* Get outer coordinate index */
        inr              = iinr[iidx];
        i_coord_offset   = DIM*inr;

        /* Load i particle coords and add shift vector */
        gmx_mm256_load_shift_and_1rvec_broadcast_pd(shiftvec+i_shift_offset,x+i_coord_offset,&ix0,&iy0,&iz0);

        fix0             = _mm256_setzero_pd();
        fiy0             = _mm256_setzero_pd();
        fiz0             = _mm256_setzero_pd();

        /* Load parameters for i particles */
        iq0              = _mm256_mul_pd(facel,_mm256_set1_pd(charge[inr+0]));

        /* Reset potential sums */
        velecsum         = _mm256_setzero_pd();

        /* Start inner kernel loop */
        for(jidx=j_index_start; jidx<j_index_end && jjnr[jidx+3]>=0; jidx+=4)
        {

            /* Get j neighbor index, and coordinate index */
            jnrA             = jjnr[jidx];
            jnrB             = jjnr[jidx+1];
            jnrC             = jjnr[jidx+2];
            jnrD             = jjnr[jidx+3];
            j_coord_offsetA  = DIM*jnrA;
            j_coord_offsetB  = DIM*jnrB;
            j_coord_offsetC  = DIM*jnrC;
            j_coord_offsetD  = DIM*jnrD;

            /* load j atom coordinates */
            gmx_mm256_load_1rvec_4ptr_swizzle_pd(x+j_coord_offsetA,x+j_coord_offsetB,
                                                 x+j_coord_offsetC,x+j_coord_offsetD,
                                                 &jx0,&jy0,&jz0);

            /* Calculate displacement vector */
            dx00             = _mm256_sub_pd(ix0,jx0);
            dy00             = _mm256_sub_pd(iy0,jy0);
            dz00             = _mm256_sub_pd(iz0,jz0);

            /* Calculate squared distance and things based on it */
            rsq00            = gmx_mm256_calc_rsq_pd(dx00,dy00,dz00);

            rinv00           = avx256_invsqrt_d(rsq00);

            rinvsq00         = _mm256_mul_pd(rinv00,rinv00);

            /* Load parameters for j particles */
            jq0              = gmx_mm256_load_4real_swizzle_pd(charge+jnrA+0,charge+jnrB+0,
                                                                 charge+jnrC+0,charge+jnrD+0);

            /**************************
             * CALCULATE INTERACTIONS *
             **************************/

            /* Compute parameters for interactions between i and j atoms */
            qq00             = _mm256_mul_pd(iq0,jq0);

            /* REACTION-FIELD ELECTROSTATICS */
            velec            = _mm256_mul_pd(qq00,_mm256_sub_pd(_mm256_add_pd(rinv00,_mm256_mul_pd(krf,rsq00)),crf));
            felec            = _mm256_mul_pd(qq00,_mm256_sub_pd(_mm256_mul_pd(rinv00,rinvsq00),krf2));

            /* Update potential sum for this i atom from the interaction with this j atom. */
            velecsum         = _mm256_add_pd(velecsum,velec);

            fscal            = felec;

            /* Calculate temporary vectorial force */
            tx               = _mm256_mul_pd(fscal,dx00);
            ty               = _mm256_mul_pd(fscal,dy00);
            tz               = _mm256_mul_pd(fscal,dz00);

            /* Update vectorial force */
            fix0             = _mm256_add_pd(fix0,tx);
            fiy0             = _mm256_add_pd(fiy0,ty);
            fiz0             = _mm256_add_pd(fiz0,tz);

            fjptrA             = f+j_coord_offsetA;
            fjptrB             = f+j_coord_offsetB;
            fjptrC             = f+j_coord_offsetC;
            fjptrD             = f+j_coord_offsetD;
            gmx_mm256_decrement_1rvec_4ptr_swizzle_pd(fjptrA,fjptrB,fjptrC,fjptrD,tx,ty,tz);

            /* Inner loop uses 32 flops */
        }

        if(jidx<j_index_end)
        {

            /* Get j neighbor index, and coordinate index */
            jnrlistA         = jjnr[jidx];
            jnrlistB         = jjnr[jidx+1];
            jnrlistC         = jjnr[jidx+2];
            jnrlistD         = jjnr[jidx+3];
            /* Sign of each element will be negative for non-real atoms.
             * This mask will be 0xFFFFFFFF for dummy entries and 0x0 for real ones,
             * so use it as val = _mm_andnot_pd(mask,val) to clear dummy entries.
             */
            tmpmask0 = gmx_mm_castsi128_ps(_mm_cmplt_epi32(_mm_loadu_si128((const __m128i *)(jjnr+jidx)),_mm_setzero_si128()));

            tmpmask1 = _mm_permute_ps(tmpmask0,_GMX_MM_PERMUTE(3,3,2,2));
            tmpmask0 = _mm_permute_ps(tmpmask0,_GMX_MM_PERMUTE(1,1,0,0));
            dummy_mask = _mm256_castps_pd(gmx_mm256_set_m128(tmpmask1,tmpmask0));

            jnrA       = (jnrlistA>=0) ? jnrlistA : 0;
            jnrB       = (jnrlistB>=0) ? jnrlistB : 0;
            jnrC       = (jnrlistC>=0) ? jnrlistC : 0;
            jnrD       = (jnrlistD>=0) ? jnrlistD : 0;
            j_coord_offsetA  = DIM*jnrA;
            j_coord_offsetB  = DIM*jnrB;
            j_coord_offsetC  = DIM*jnrC;
            j_coord_offsetD  = DIM*jnrD;

            /* load j atom coordinates */
            gmx_mm256_load_1rvec_4ptr_swizzle_pd(x+j_coord_offsetA,x+j_coord_offsetB,
                                                 x+j_coord_offsetC,x+j_coord_offsetD,
                                                 &jx0,&jy0,&jz0);

            /* Calculate displacement vector */
            dx00             = _mm256_sub_pd(ix0,jx0);
            dy00             = _mm256_sub_pd(iy0,jy0);
            dz00             = _mm256_sub_pd(iz0,jz0);

            /* Calculate squared distance and things based on it */
            rsq00            = gmx_mm256_calc_rsq_pd(dx00,dy00,dz00);

            rinv00           = avx256_invsqrt_d(rsq00);

            rinvsq00         = _mm256_mul_pd(rinv00,rinv00);

            /* Load parameters for j particles */
            jq0              = gmx_mm256_load_4real_swizzle_pd(charge+jnrA+0,charge+jnrB+0,
                                                                 charge+jnrC+0,charge+jnrD+0);

            /**************************
             * CALCULATE INTERACTIONS *
             **************************/

            /* Compute parameters for interactions between i and j atoms */
            qq00             = _mm256_mul_pd(iq0,jq0);

            /* REACTION-FIELD ELECTROSTATICS */
            velec            = _mm256_mul_pd(qq00,_mm256_sub_pd(_mm256_add_pd(rinv00,_mm256_mul_pd(krf,rsq00)),crf));
            felec            = _mm256_mul_pd(qq00,_mm256_sub_pd(_mm256_mul_pd(rinv00,rinvsq00),krf2));

            /* Update potential sum for this i atom from the interaction with this j atom. */
            velec            = _mm256_andnot_pd(dummy_mask,velec);
            velecsum         = _mm256_add_pd(velecsum,velec);

            fscal            = felec;

            fscal            = _mm256_andnot_pd(dummy_mask,fscal);

            /* Calculate temporary vectorial force */
            tx               = _mm256_mul_pd(fscal,dx00);
            ty               = _mm256_mul_pd(fscal,dy00);
            tz               = _mm256_mul_pd(fscal,dz00);

            /* Update vectorial force */
            fix0             = _mm256_add_pd(fix0,tx);
            fiy0             = _mm256_add_pd(fiy0,ty);
            fiz0             = _mm256_add_pd(fiz0,tz);

            fjptrA             = (jnrlistA>=0) ? f+j_coord_offsetA : scratch;
            fjptrB             = (jnrlistB>=0) ? f+j_coord_offsetB : scratch;
            fjptrC             = (jnrlistC>=0) ? f+j_coord_offsetC : scratch;
            fjptrD             = (jnrlistD>=0) ? f+j_coord_offsetD : scratch;
            gmx_mm256_decrement_1rvec_4ptr_swizzle_pd(fjptrA,fjptrB,fjptrC,fjptrD,tx,ty,tz);

            /* Inner loop uses 32 flops */
        }

        /* End of innermost loop */

        gmx_mm256_update_iforce_1atom_swizzle_pd(fix0,fiy0,fiz0,
                                                 f+i_coord_offset,fshift+i_shift_offset);

        ggid                        = gid[iidx];
        /* Update potential energies */
        gmx_mm256_update_1pot_pd(velecsum,kernel_data->energygrp_elec+ggid);

        /* Increment number of inner iterations */
        inneriter                  += j_index_end - j_index_start;

        /* Outer loop uses 8 flops */
    }

    /* Increment number of outer iterations */
    outeriter        += nri;

    /* Update outer/inner flops */

    inc_nrnb(nrnb,eNR_NBKERNEL_ELEC_VF,outeriter*8 + inneriter*32);
}
/*
 * Gromacs nonbonded kernel:   nb_kernel_ElecRF_VdwNone_GeomP1P1_F_avx_256_double
 * Electrostatics interaction: ReactionField
 * VdW interaction:            None
 * Geometry:                   Particle-Particle
 * Calculate force/pot:        Force
 */
void
nb_kernel_ElecRF_VdwNone_GeomP1P1_F_avx_256_double
                    (t_nblist                    * gmx_restrict       nlist,
                     rvec                        * gmx_restrict          xx,
                     rvec                        * gmx_restrict          ff,
                     struct t_forcerec           * gmx_restrict          fr,
                     t_mdatoms                   * gmx_restrict     mdatoms,
                     nb_kernel_data_t gmx_unused * gmx_restrict kernel_data,
                     t_nrnb                      * gmx_restrict        nrnb)
{
    /* Suffixes 0,1,2,3 refer to particle indices for waters in the inner or outer loop, or 
     * just 0 for non-waters.
     * Suffixes A,B,C,D refer to j loop unrolling done with AVX, e.g. for the four different
     * jnr indices corresponding to data put in the four positions in the SIMD register.
     */
    int              i_shift_offset,i_coord_offset,outeriter,inneriter;
    int              j_index_start,j_index_end,jidx,nri,inr,ggid,iidx;
    int              jnrA,jnrB,jnrC,jnrD;
    int              jnrlistA,jnrlistB,jnrlistC,jnrlistD;
    int              jnrlistE,jnrlistF,jnrlistG,jnrlistH;
    int              j_coord_offsetA,j_coord_offsetB,j_coord_offsetC,j_coord_offsetD;
    int              *iinr,*jindex,*jjnr,*shiftidx,*gid;
    real             rcutoff_scalar;
    real             *shiftvec,*fshift,*x,*f;
    real             *fjptrA,*fjptrB,*fjptrC,*fjptrD;
    real             scratch[4*DIM];
    __m256d          tx,ty,tz,fscal,rcutoff,rcutoff2,jidxall;
    real *           vdwioffsetptr0;
    __m256d          ix0,iy0,iz0,fix0,fiy0,fiz0,iq0,isai0;
    int              vdwjidx0A,vdwjidx0B,vdwjidx0C,vdwjidx0D;
    __m256d          jx0,jy0,jz0,fjx0,fjy0,fjz0,jq0,isaj0;
    __m256d          dx00,dy00,dz00,rsq00,rinv00,rinvsq00,r00,qq00,c6_00,c12_00;
    __m256d          velec,felec,velecsum,facel,crf,krf,krf2;
    real             *charge;
    __m256d          dummy_mask,cutoff_mask;
    __m128           tmpmask0,tmpmask1;
    __m256d          signbit = _mm256_castsi256_pd( _mm256_set1_epi32(0x80000000) );
    __m256d          one     = _mm256_set1_pd(1.0);
    __m256d          two     = _mm256_set1_pd(2.0);
    x                = xx[0];
    f                = ff[0];

    nri              = nlist->nri;
    iinr             = nlist->iinr;
    jindex           = nlist->jindex;
    jjnr             = nlist->jjnr;
    shiftidx         = nlist->shift;
    gid              = nlist->gid;
    shiftvec         = fr->shift_vec[0];
    fshift           = fr->fshift[0];
    facel            = _mm256_set1_pd(fr->ic->epsfac);
    charge           = mdatoms->chargeA;
    krf              = _mm256_set1_pd(fr->ic->k_rf);
    krf2             = _mm256_set1_pd(fr->ic->k_rf*2.0);
    crf              = _mm256_set1_pd(fr->ic->c_rf);

    /* Avoid stupid compiler warnings */
    jnrA = jnrB = jnrC = jnrD = 0;
    j_coord_offsetA = 0;
    j_coord_offsetB = 0;
    j_coord_offsetC = 0;
    j_coord_offsetD = 0;

    outeriter        = 0;
    inneriter        = 0;

    for(iidx=0;iidx<4*DIM;iidx++)
    {
        scratch[iidx] = 0.0;
    }

    /* Start outer loop over neighborlists */
    for(iidx=0; iidx<nri; iidx++)
    {
        /* Load shift vector for this list */
        i_shift_offset   = DIM*shiftidx[iidx];

        /* Load limits for loop over neighbors */
        j_index_start    = jindex[iidx];
        j_index_end      = jindex[iidx+1];

        /* Get outer coordinate index */
        inr              = iinr[iidx];
        i_coord_offset   = DIM*inr;

        /* Load i particle coords and add shift vector */
        gmx_mm256_load_shift_and_1rvec_broadcast_pd(shiftvec+i_shift_offset,x+i_coord_offset,&ix0,&iy0,&iz0);

        fix0             = _mm256_setzero_pd();
        fiy0             = _mm256_setzero_pd();
        fiz0             = _mm256_setzero_pd();

        /* Load parameters for i particles */
        iq0              = _mm256_mul_pd(facel,_mm256_set1_pd(charge[inr+0]));

        /* Start inner kernel loop */
        for(jidx=j_index_start; jidx<j_index_end && jjnr[jidx+3]>=0; jidx+=4)
        {

            /* Get j neighbor index, and coordinate index */
            jnrA             = jjnr[jidx];
            jnrB             = jjnr[jidx+1];
            jnrC             = jjnr[jidx+2];
            jnrD             = jjnr[jidx+3];
            j_coord_offsetA  = DIM*jnrA;
            j_coord_offsetB  = DIM*jnrB;
            j_coord_offsetC  = DIM*jnrC;
            j_coord_offsetD  = DIM*jnrD;

            /* load j atom coordinates */
            gmx_mm256_load_1rvec_4ptr_swizzle_pd(x+j_coord_offsetA,x+j_coord_offsetB,
                                                 x+j_coord_offsetC,x+j_coord_offsetD,
                                                 &jx0,&jy0,&jz0);

            /* Calculate displacement vector */
            dx00             = _mm256_sub_pd(ix0,jx0);
            dy00             = _mm256_sub_pd(iy0,jy0);
            dz00             = _mm256_sub_pd(iz0,jz0);

            /* Calculate squared distance and things based on it */
            rsq00            = gmx_mm256_calc_rsq_pd(dx00,dy00,dz00);

            rinv00           = avx256_invsqrt_d(rsq00);

            rinvsq00         = _mm256_mul_pd(rinv00,rinv00);

            /* Load parameters for j particles */
            jq0              = gmx_mm256_load_4real_swizzle_pd(charge+jnrA+0,charge+jnrB+0,
                                                                 charge+jnrC+0,charge+jnrD+0);

            /**************************
             * CALCULATE INTERACTIONS *
             **************************/

            /* Compute parameters for interactions between i and j atoms */
            qq00             = _mm256_mul_pd(iq0,jq0);

            /* REACTION-FIELD ELECTROSTATICS */
            felec            = _mm256_mul_pd(qq00,_mm256_sub_pd(_mm256_mul_pd(rinv00,rinvsq00),krf2));

            fscal            = felec;

            /* Calculate temporary vectorial force */
            tx               = _mm256_mul_pd(fscal,dx00);
            ty               = _mm256_mul_pd(fscal,dy00);
            tz               = _mm256_mul_pd(fscal,dz00);

            /* Update vectorial force */
            fix0             = _mm256_add_pd(fix0,tx);
            fiy0             = _mm256_add_pd(fiy0,ty);
            fiz0             = _mm256_add_pd(fiz0,tz);

            fjptrA             = f+j_coord_offsetA;
            fjptrB             = f+j_coord_offsetB;
            fjptrC             = f+j_coord_offsetC;
            fjptrD             = f+j_coord_offsetD;
            gmx_mm256_decrement_1rvec_4ptr_swizzle_pd(fjptrA,fjptrB,fjptrC,fjptrD,tx,ty,tz);

            /* Inner loop uses 27 flops */
        }

        if(jidx<j_index_end)
        {

            /* Get j neighbor index, and coordinate index */
            jnrlistA         = jjnr[jidx];
            jnrlistB         = jjnr[jidx+1];
            jnrlistC         = jjnr[jidx+2];
            jnrlistD         = jjnr[jidx+3];
            /* Sign of each element will be negative for non-real atoms.
             * This mask will be 0xFFFFFFFF for dummy entries and 0x0 for real ones,
             * so use it as val = _mm_andnot_pd(mask,val) to clear dummy entries.
             */
            tmpmask0 = gmx_mm_castsi128_ps(_mm_cmplt_epi32(_mm_loadu_si128((const __m128i *)(jjnr+jidx)),_mm_setzero_si128()));

            tmpmask1 = _mm_permute_ps(tmpmask0,_GMX_MM_PERMUTE(3,3,2,2));
            tmpmask0 = _mm_permute_ps(tmpmask0,_GMX_MM_PERMUTE(1,1,0,0));
            dummy_mask = _mm256_castps_pd(gmx_mm256_set_m128(tmpmask1,tmpmask0));

            jnrA       = (jnrlistA>=0) ? jnrlistA : 0;
            jnrB       = (jnrlistB>=0) ? jnrlistB : 0;
            jnrC       = (jnrlistC>=0) ? jnrlistC : 0;
            jnrD       = (jnrlistD>=0) ? jnrlistD : 0;
            j_coord_offsetA  = DIM*jnrA;
            j_coord_offsetB  = DIM*jnrB;
            j_coord_offsetC  = DIM*jnrC;
            j_coord_offsetD  = DIM*jnrD;

            /* load j atom coordinates */
            gmx_mm256_load_1rvec_4ptr_swizzle_pd(x+j_coord_offsetA,x+j_coord_offsetB,
                                                 x+j_coord_offsetC,x+j_coord_offsetD,
                                                 &jx0,&jy0,&jz0);

            /* Calculate displacement vector */
            dx00             = _mm256_sub_pd(ix0,jx0);
            dy00             = _mm256_sub_pd(iy0,jy0);
            dz00             = _mm256_sub_pd(iz0,jz0);

            /* Calculate squared distance and things based on it */
            rsq00            = gmx_mm256_calc_rsq_pd(dx00,dy00,dz00);

            rinv00           = avx256_invsqrt_d(rsq00);

            rinvsq00         = _mm256_mul_pd(rinv00,rinv00);

            /* Load parameters for j particles */
            jq0              = gmx_mm256_load_4real_swizzle_pd(charge+jnrA+0,charge+jnrB+0,
                                                                 charge+jnrC+0,charge+jnrD+0);

            /**************************
             * CALCULATE INTERACTIONS *
             **************************/

            /* Compute parameters for interactions between i and j atoms */
            qq00             = _mm256_mul_pd(iq0,jq0);

            /* REACTION-FIELD ELECTROSTATICS */
            felec            = _mm256_mul_pd(qq00,_mm256_sub_pd(_mm256_mul_pd(rinv00,rinvsq00),krf2));

            fscal            = felec;

            fscal            = _mm256_andnot_pd(dummy_mask,fscal);

            /* Calculate temporary vectorial force */
            tx               = _mm256_mul_pd(fscal,dx00);
            ty               = _mm256_mul_pd(fscal,dy00);
            tz               = _mm256_mul_pd(fscal,dz00);

            /* Update vectorial force */
            fix0             = _mm256_add_pd(fix0,tx);
            fiy0             = _mm256_add_pd(fiy0,ty);
            fiz0             = _mm256_add_pd(fiz0,tz);

            fjptrA             = (jnrlistA>=0) ? f+j_coord_offsetA : scratch;
            fjptrB             = (jnrlistB>=0) ? f+j_coord_offsetB : scratch;
            fjptrC             = (jnrlistC>=0) ? f+j_coord_offsetC : scratch;
            fjptrD             = (jnrlistD>=0) ? f+j_coord_offsetD : scratch;
            gmx_mm256_decrement_1rvec_4ptr_swizzle_pd(fjptrA,fjptrB,fjptrC,fjptrD,tx,ty,tz);

            /* Inner loop uses 27 flops */
        }

        /* End of innermost loop */

        gmx_mm256_update_iforce_1atom_swizzle_pd(fix0,fiy0,fiz0,
                                                 f+i_coord_offset,fshift+i_shift_offset);

        /* Increment number of inner iterations */
        inneriter                  += j_index_end - j_index_start;

        /* Outer loop uses 7 flops */
    }

    /* Increment number of outer iterations */
    outeriter        += nri;

    /* Update outer/inner flops */

    inc_nrnb(nrnb,eNR_NBKERNEL_ELEC_F,outeriter*7 + inneriter*27);
}
