*======================================================================*
C P.Moreau - Sept. 2020
*======================================================================*
      
*======================================================================*
      MODULE define_arrays
*======================================================================*
      real, allocatable, save, dimension (:) :: particle_mass, 
     & particle_baryon,list_b_all
      integer, allocatable, save, dimension (:) :: particle_ID
      character*10, allocatable, save, dimension (:) :: particle_name
      character*100, allocatable, save, dimension (:) :: list_paths
      integer, allocatable, save, dimension (:) :: N_events_b
      integer, allocatable, save, dimension (:,:) :: N_part_b
      real, allocatable, save, dimension (:,:,:) :: dNdeta_b,dNdy_b,
     & mean_pT_b,dNBBBARdy_y
      real, allocatable, save, dimension (:) :: ev_dNdeta,ev_dNdy,
     & ev_dNBBBARdy_y
      integer, allocatable, save, dimension (:,:) :: N_mean_pT_b
      real, allocatable, save, dimension (:,:,:,:) :: dNdeta_eta,dNdy_y,
     & dNdpT_pT,dNdmT_mT
      real, allocatable, save, dimension (:,:) :: ev_dNdeta_eta,
     & ev_dNdy_y,ev_dNdpT_pT,ev_dNdmT_mT
      
      CONTAINS
*======================================================================*
      SUBROUTINE ALLOCATE_particles(N_particles)
      integer, intent(out) :: N_particles

      OPEN(5,FILE='./particle_info.dat',STATUS='OLD')
      READ(5,*) N_particles

      ! + charged particles
      N_particles = N_particles + 1
      
      allocate(particle_ID(1:N_particles))
      allocate(particle_name(1:N_particles))
      allocate(particle_mass(1:N_particles))
      allocate(particle_baryon(1:N_particles))
      
      DO ipart=1,N_particles-1
          READ(5,*) i,particle_ID(ipart),particle_name(ipart),
     & particle_mass(ipart),particle_baryon(ipart)
      END DO
      CLOSE(5)
      ! Add an entry for charged particles
      particle_name(N_particles) = 'ch'
      END SUBROUTINE

*======================================================================*
      SUBROUTINE ALLOCATE_paths(N_paths)
      integer, intent(out) :: N_paths

      OPEN(5,FILE='./path_phsd.dat',STATUS='OLD')
      READ(5,*) N_paths
      
      allocate(list_paths(1:N_paths))
      
      DO ipath=1,N_paths
          READ(5,'(A100)') list_paths(ipath)
      END DO
      CLOSE(5)
      END SUBROUTINE

*======================================================================*
      SUBROUTINE ALLOCATE_Npart(Nb,N_particles)
      allocate(list_b_all(1:Nb))
      allocate(N_events_b(1:Nb))
      allocate(N_part_b(1:Nb,1:2))
      allocate(dNdeta_b(1:Nb,1:N_particles,1:2))
      allocate(dNdy_b(1:Nb,1:N_particles,1:2))
      allocate(mean_pT_b(1:Nb,1:N_particles,1:2))
      allocate(N_mean_pT_b(1:Nb,1:N_particles))
      allocate(ev_dNdeta(1:N_particles))
      allocate(ev_dNdy(1:N_particles))
      N_events_b(:) = 0
      N_part_b(:,:) = 0
      dNdeta_b(:,:,:) = 0.
      dNdy_b(:,:,:) = 0.
      mean_pT_b(:,:,:) = 0.
      N_mean_pT_b(:,:) = 0
      END SUBROUTINE
*======================================================================*
      SUBROUTINE ALLOCATE_eta(Nb,N_particles,Neta)
      allocate(dNdeta_eta(1:Nb,1:N_particles,-Neta:Neta,1:2))
      allocate(ev_dNdeta_eta(1:N_particles,-Neta:Neta))
      dNdeta_eta(:,:,:,:) = 0.
      END SUBROUTINE
*======================================================================*
      SUBROUTINE ALLOCATE_y(Nb,N_particles,Ny)
      allocate(dNdy_y(1:Nb,1:N_particles,-Ny:Ny,1:2))
      allocate(dNBBBARdy_y(1:Nb,-Ny:Ny,1:2))
      allocate(ev_dNdy_y(1:N_particles,-Ny:Ny))
      allocate(ev_dNBBBARdy_y(-Ny:Ny))
      dNdy_y(:,:,:,:) = 0.
      dNBBBARdy_y(:,:,:) = 0.
      END SUBROUTINE
*======================================================================*
      SUBROUTINE ALLOCATE_pT(Nb,N_particles,NpT)
      allocate(dNdpT_pT(1:Nb,1:N_particles,0:NpT,1:2))
      allocate(ev_dNdpT_pT(1:N_particles,0:NpT))
      dNdpT_pT(:,:,:,:) = 0.
      END SUBROUTINE
*======================================================================*
      SUBROUTINE ALLOCATE_mT(Nb,N_particles,NmT)
      allocate(dNdmT_mT(1:Nb,1:N_particles,0:NmT,1:2))
      allocate(ev_dNdmT_mT(1:N_particles,0:NmT))
      dNdmT_mT(:,:,:,:) = 0.
      END SUBROUTINE
*======================================================================*

      END MODULE
  
*======================================================================*
      PROGRAM READ_PHSD
*======================================================================*
     
      USE define_arrays
      real :: midrapy,midrapeta,mTbin,mTmax,mT,mass
      logical :: count
      character(len=10) :: string_b
      real, parameter :: pi=4.*ATAN(1.)
      
      ! read information about the analysis
      OPEN(4,FILE='./inputf.dat',STATUS='OLD')

      READ(4,*) BMIN
      READ(4,*) xBMIN
      READ(4,*) BMAX
      READ(4,*) xBMAX
      READ(4,*) DBIMP
      READ(4,*) ISUBS
      READ(4,*) NUM
      READ(4,*) MASSTA
      READ(4,*) MASSPR
      READ(4,*) y_proj
      READ(4,*) midrapy
      READ(4,*) midrapeta
      READ(4,*) ybin
      READ(4,*) ylim
      READ(4,*) etabin
      READ(4,*) etalim
      READ(4,*) pTbin
      READ(4,*) pTmax
      READ(4,*) mTbin
      READ(4,*) mTmax
      CLOSE(4)

      ! bin at midrapidity
      deta = 2.*midrapeta
      dy = 2.*midrapy

      ! number of impact parameters to read
      Nb_tot = nint((BMAX-BMIN)/DBIMP+1)
      ! number of impact parameters to record
      Nb  = nint((xBMAX-xBMIN)/DBIMP+1)
      ! number of bin in eta, y, pT, mT
      Neta = nint(etalim/etabin)
      Ny = nint(ylim/ybin)
      NpT = nint(pTmax/pTbin)-1 ! -1 because array starts at 0
      NmT = nint(mTmax/mTbin)-1

      ! load particle and file information
      CALL ALLOCATE_particles(N_particles)
      CALL ALLOCATE_paths(N_paths)

      ! initialize all arrays
      CALL ALLOCATE_Npart(Nb,N_particles)
      CALL ALLOCATE_eta(Nb,N_particles,Neta)
      CALL ALLOCATE_y(Nb,N_particles,Ny)
      CALL ALLOCATE_pT(Nb,N_particles,NpT)
      CALL ALLOCATE_mT(Nb,N_particles,NmT)

      ! array containing the values of impact parameters
      DO ib=1,Nb
        list_b_all(ib) = BMIN+(ib-1)*DBIMP
      END DO

      ! number in the array for charged particles
      iPch=N_particles

      ! format to read particle information in phsd.dat
 124  FORMAT(1I6,3X,2I6,1E16.8,1I6)      ! header 1
 125  FORMAT(1I6,8E16.8)                 ! header 2
 126  FORMAT(1X,1I8,1I6,4E16.8,3X,1I6)   ! particle info

      ! format to display info
 127  FORMAT('   - ISUB = ',I4,' ; B = ',1F4.1,' fm')
 128  FORMAT('   - ISUB = ',I4,' ; B = ',1F4.1,' fm (not counted')

      ! loop over files
      DO ifile=1,N_paths
            
        write(*,*) 'Reading:',list_paths(ifile)
        OPEN(3000,FILE=list_paths(ifile),STATUS='OLD')

        ! loop over impact parameter
        DO ib=1,Nb_tot

          ! skip if B > xBMAX
          if(list_b_all(ib).gt.xBMAX) then
            GO TO 30
          end if

          ! loop over ISUBS
          DO iISUB=1,ISUBS

            ! Display the current ISUB
            if(list_b_all(ib).lt.xBMIN) then
                write(*,128) iISUB,list_b_all(ib)
            else
                write(*,127) iISUB,list_b_all(ib)
            end if

            ! loop over NUM
            DO iNUM=1,NUM

              ! initialize arrays for this event
              ev_dNdeta(:)=0.
              ev_dNdy(:)=0.
              ev_dNdeta_eta(:,:)=0.
              ev_dNdy_y(:,:)=0.
              ev_dNdpT_pT(:,:)=0.
              ev_dNdmT_mT(:,:)=0.
              ev_dNBBBARdy_y(:)=0.

              READ(3000,124,END=30) N, ISUB, IRUN, BIMP, IBweight
              READ(3000,125,END=30) Np, phi2, epsi2, phi3, epsi3, 
     &                         phi4, epsi4, phi5, epsi5   

              ! count this event?
              count=.False.
              ! only count inelastic events
              if(N.gt.(MASSTA+MASSPR)) then
                ! save number of participants
                N_part_b(ib,1)=N_part_b(ib,1)+Np
                N_part_b(ib,2)=N_part_b(ib,2)+Np**2.
                N_events_b(ib)=N_events_b(ib)+1
                count=.True.
              end if
                 
              ! loop over particles
              DO ipart=1,N
                READ(3000,126,END=30) ID,IDQ,PX,PY,PZ,P0,iHist
                ! if elastic event, don't count
                if(.not.count) CYCLE
                ! If Bmin > Bimp don't count the particles of this event
                if((xBMIN-BIMP).gt.0.01) CYCLE               

                ! Identify the particle from KF code
                DO IP=1,N_particles
                  if(particle_ID(IP).eq.ID) GO TO 7                   
                END DO  
                ! If the particle ID is not recognized, don't count this particle
                CYCLE                                            
 7              CONTINUE

                mass = particle_mass(IP)

                y=0.5*log((P0+PZ)/(P0-PZ))

                if(IP.eq.8 .or. IP.eq.10) then
                ! if |y| > yproj-0.5, don't count
                ! 0.5 is arbitrary, but works fine
                  if(iHist.eq.-1 .and. abs(y).gt.(y_proj-0.5)) CYCLE
                end if

                ! Count Sigma0 with Lambda
                if(IP.eq.14) then
                  IP=12
                ! Count Sigma~0 with Lambda~
                else if(IP.eq.15) then
                  IP=13
                end if

                eta=0.5*log((sqrt(PX**2+PY**2+PZ**2)+PZ) 
     &                     /(sqrt(PX**2+PY**2+PZ**2)-PZ))
                pT=sqrt(PX**2+PY**2)
                mT=sqrt(pT**2.+mass**2.)
                phi = atan2(PY,PX)

                ! calculate observables for this event
                if(abs(eta).le.midrapeta) then
                  ! dNdeta
                  ev_dNdeta(IP)=ev_dNdeta(IP)+1./deta
                  if(IDQ.ne.0) then
                    ev_dNdeta(IPch)=ev_dNdeta(IPch)+1./deta
                  end if
                end if
                if(abs(y).le.midrapy) then
                  ! dNdy
                  ev_dNdy(IP)=ev_dNdy(IP)+1./dy
                  if(IDQ.ne.0) then
                      ev_dNdy(IPch)=ev_dNdy(IPch)+1./dy
                  end if
                  ! mean_pT at midrapidity
                  mean_pT_b(ib,IP,1)=mean_pT_b(ib,IP,1)+pT
                  mean_pT_b(ib,IP,2)=mean_pT_b(ib,IP,2)+pT**2.
                  N_mean_pT_b(ib,IP)=N_mean_pT_b(ib,IP)+1
                  if(IDQ.ne.0) then
                    mean_pT_b(ib,IPch,1)=mean_pT_b(ib,IPch,1)+pT
                    mean_pT_b(ib,IPch,2)=mean_pT_b(ib,IPch,2)+pT**2.
                    N_mean_pT_b(ib,IPch)=N_mean_pT_b(ib,IPch)+1
                  end if
                  ! pT spectra
                  ipT=int(pT/pTbin)
                  if(ipT.le.NpT) then
                    ev_dNdpT_pT(IP,ipT)=ev_dNdpT_pT(IP,ipT)
     &                                  +1./(2.*pi*pT*dy*pTbin)
                    if(IDQ.ne.0) then
                      ev_dNdpT_pT(IPch,ipT)=ev_dNdpT_pT(IPch,ipT)
     &                                  +1./(2.*pi*pT*dy*pTbin)
                    end if
                  end if
                  ! mT spectra
                  imT=int((mT-mass)/mTbin)
                  if(imT.le.NmT) then
                    ev_dNdmT_mT(IP,imT)=ev_dNdmT_mT(IP,imT)+
     &                                  +1./(2.*pi*mT*dy*mTbin)
                  end if
                end if
                ! y spectra
                iy=nint(y/ybin)
                if(abs(iy).le.Ny) then
                  ev_dNdy_y(IP,iy)=ev_dNdy_y(IP,iy)+1./ybin
                  ev_dNBBBARdy_y(iy)=ev_dNBBBARdy_y(iy)
     &                                  +particle_baryon(IP)/ybin
                  if(IDQ.ne.0) then
                    ev_dNdy_y(IPch,iy)=ev_dNdy_y(IPch,iy)+1./ybin
                  end if
                end if
                ! eta spectra
                ieta=nint(eta/etabin)
                if(abs(ieta).le.Neta) then
                  ev_dNdeta_eta(IP,ieta)=ev_dNdeta_eta(IP,ieta)
     &                                   +1./etabin
                  if(IDQ.ne.0) then
                    ev_dNdeta_eta(IPch,ieta)=ev_dNdeta_eta(IPch,ieta)
     &                                       +1./etabin
                  end if
                end if

              END DO ! loop over particles

              ! add this event observables to final observables
              DO IP=1,N_particles
                ! add dNdeta
                dNdeta_b(ib,IP,1)=dNdeta_b(ib,IP,1)+ev_dNdeta(IP)
                dNdeta_b(ib,IP,2)=dNdeta_b(ib,IP,2)+ev_dNdeta(IP)**2.
                ! add dNdy
                dNdy_b(ib,IP,1)=dNdy_b(ib,IP,1)+ev_dNdy(IP)
                dNdy_b(ib,IP,2)=dNdy_b(ib,IP,2)+ev_dNdy(IP)**2.
                ! add dNdpT_pT
                DO ipT=0,NpT
                  dNdpT_pT(ib,IP,ipT,1)=dNdpT_pT(ib,IP,ipT,1)
     &                                  +ev_dNdpT_pT(IP,ipT)
                  dNdpT_pT(ib,IP,ipT,2)=dNdpT_pT(ib,IP,ipT,2)
     &                                  +ev_dNdpT_pT(IP,ipT)**2.
                END DO
                ! add dNdpT_pT
                DO imT=0,NmT
                  dNdmT_mT(ib,IP,imT,1)=dNdmT_mT(ib,IP,imT,1)
     &                                  +ev_dNdmT_mT(IP,imT)
                  dNdmT_mT(ib,IP,imT,2)=dNdmT_mT(ib,IP,imT,2)
     &                                  +ev_dNdmT_mT(IP,imT)**2.
                END DO
                ! add dNdeta_eta
                DO ieta=-Neta,Neta
                  dNdeta_eta(ib,IP,ieta,1)=dNdeta_eta(ib,IP,ieta,1)
     &                   +ev_dNdeta_eta(IP,ieta)
                  dNdeta_eta(ib,IP,ieta,2)=dNdeta_eta(ib,IP,ieta,2)
     &                   +ev_dNdeta_eta(IP,ieta)**2.
                END DO
                ! add dNdy_y
                DO iy=-Ny,Ny
                  dNdy_y(ib,IP,iy,1)=dNdy_y(ib,IP,iy,1)
     &                   +ev_dNdy_y(IP,iy)
                  dNdy_y(ib,IP,iy,2)=dNdy_y(ib,IP,iy,2)
     &                   +ev_dNdy_y(IP,iy)**2.
                END DO

              END DO ! loop over particles for final observables
              DO iy=-Ny,Ny
                dNBBBARdy_y(ib,iy,1)=dNBBBARdy_y(ib,iy,1)
     &                   +ev_dNBBBARdy_y(iy)
                dNBBBARdy_y(ib,iy,2)=dNBBBARdy_y(ib,iy,2)
     &                   +ev_dNBBBARdy_y(iy)**2.
              END DO

            END DO ! loop over NUM
          END DO ! loop over ISUBS
        END DO ! loop over impact parameter
 30     CONTINUE ! End of file
        CLOSE(3000)
      END DO ! loop over files
      
      ! Output final observables
      write(*,*) 'Creating output'
      
      DO ib=1,Nb
        CALL b_to_string(list_b_all(ib),string_b)
        OPEN(4,FILE='./Nevents_b'//trim(string_b)
     &               //'.dat',STATUS='NEW')
        OPEN(5,FILE='./Npart_b'//trim(string_b)
     &               //'.dat',STATUS='NEW')
          WRITE(4,*) N_events_b(ib)
          WRITE(5,*) N_part_b(ib,1),N_part_b(ib,2)
        CLOSE(4)
        CLOSE(5)
      END DO

      DO ib=1,Nb
        CALL b_to_string(list_b_all(ib),string_b)
        DO IP=1,N_particles
          if(IP.eq.14 .or. IP.eq.15) CYCLE ! Skip Sigma0 & Sigma~0
          OPEN(4,FILE='./dNdy_b'//trim(string_b)
     &              //'_'//trim(particle_name(IP))//'.dat',STATUS='NEW')
          OPEN(5,FILE='./dNdeta_b'//trim(string_b)
     &              //'_'//trim(particle_name(IP))//'.dat',STATUS='NEW')
          OPEN(6,FILE='./mean_pT_b'//trim(string_b)
     &              //'_'//trim(particle_name(IP))//'.dat',STATUS='NEW')
          OPEN(7,FILE='./N_mean_pT_b'//trim(string_b)
     &              //'_'//trim(particle_name(IP))//'.dat',STATUS='NEW')
          
            WRITE(4,*) dNdy_b(ib,IP,1),dNdy_b(ib,IP,2)
            WRITE(5,*) dNdeta_b(ib,IP,1),dNdeta_b(ib,IP,2)
            WRITE(6,*) mean_pT_b(ib,IP,1),mean_pT_b(ib,IP,2)
            WRITE(7,*) N_mean_pT_b(ib,IP)
        END DO
        CLOSE(4)
        CLOSE(5)
        CLOSE(6)
        CLOSE(7)
      END DO

      DO ib=1,Nb
        CALL b_to_string(list_b_all(ib),string_b)
        DO IP=1,N_particles
          if(IP.eq.14 .or. IP.eq.15) CYCLE ! Skip Sigma0 & Sigma~0
          OPEN(4,FILE='./dNdpT_pT_b'//trim(string_b)
     &              //'_'//trim(particle_name(IP))//'.dat',STATUS='NEW')
          OPEN(5,FILE='./dNdmT_mT_b'//trim(string_b)
     &              //'_'//trim(particle_name(IP))//'.dat',STATUS='NEW')
          OPEN(6,FILE='./dNdeta_eta_b'//trim(string_b)
     &              //'_'//trim(particle_name(IP))//'.dat',STATUS='NEW')
          OPEN(7,FILE='./dNdy_y_b'//trim(string_b)
     &              //'_'//trim(particle_name(IP))//'.dat',STATUS='NEW')
          DO ipT=0,NpT
            WRITE(4,*) dNdpT_pT(ib,IP,ipT,1),dNdpT_pT(ib,IP,ipT,2)
          END DO
          DO imT=0,NmT
            WRITE(5,*) dNdmT_mT(ib,IP,imT,1),dNdmT_mT(ib,IP,imT,2)
          END DO
          DO ieta=-Neta,Neta
            WRITE(6,*) dNdeta_eta(ib,IP,ieta,1),dNdeta_eta(ib,IP,ieta,2)
          END DO
          DO iy=-Ny,Ny
            WRITE(7,*) dNdy_y(ib,IP,iy,1),dNdy_y(ib,IP,iy,2)
          END DO
          CLOSE(4)
          CLOSE(5)
          CLOSE(6)
          CLOSE(7)
        END DO
        OPEN(8,FILE='./dNBBBARdy_y_b'//trim(string_b)
     &              //'.dat',STATUS='NEW')
          DO iy=-Ny,Ny
            WRITE(8,*) dNBBBARdy_y(ib,iy,1),dNBBBARdy_y(ib,iy,2)
          END DO
        CLOSE(8)
      END DO

      STOP
      END PROGRAM READ_PHSD


*======================================================================*
      SUBROUTINE b_to_string(xb,string_b)
      real, intent(in) :: xb
      character(len=10), intent(out) :: string_b

      if(xb.eq.0.) then
        write(string_b,'(I1)') 0
        string_b='00'//string_b
      else if(xb.lt.1.) then
        write(string_b,'(I1)') int(10*xb)
        string_b='00'//string_b
      else if(xb.lt.10.) then
        write(string_b,'(I2)') int(10.*xb)
        string_b='0'//string_b
      else
        write(string_b,'(I3)') int(10.*xb)
      end if

      END SUBROUTINE