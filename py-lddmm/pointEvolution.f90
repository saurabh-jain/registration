
subroutine kernelMatrixLaplacianPrecompute(x, sig, ord, num_nodes, &
     u, K, K_diff, K_diff2)
  implicit none
  integer :: num_nodes
  real(8) :: x(num_nodes, 3)
  real(8) :: sig
  integer :: ord
  real(8) :: u(num_nodes, num_nodes)
  real(8) :: K(num_nodes, num_nodes)
  real(8) :: K_diff(num_nodes, num_nodes)
  real(8) :: K_diff2(num_nodes, num_nodes)

  !f2py integer, intent(in) :: num_nodes
  !f2py real(8), intent(in) :: sig
  !f2py integer, intent(in) :: ord
  !f2py real(8), intent(out), dimension(num_nodes, num_nodes) :: u
  !f2py real(8), intent(out), dimension(num_nodes, num_nodes) :: K 
  !f2py real(8), intent(out), dimension(num_nodes, num_nodes) :: K_diff 
  !f2py real(8), intent(out), dimension(num_nodes, num_nodes) :: K_diff2 

  real(8) :: ut
  real(8) :: lpt
  integer :: i,j
  real(8) :: c_(5, 5), c1_(4, 4), c2_(3,3)
  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
  c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
       0.,0.,0., 1./105 /), (/4,4/))
  c2_ = reshape((/ 1.0/3, 1./15, 1./35,    0., 1./15, 1./35,   0.,0., 1./105 /), (/3,3/))

  !$omp parallel do private(j,i,ut,lpt) shared (u,K,K_diff,K_diff2,c_,c1_,c2_)
  do j = 1, num_nodes, 1
     do i = 1, num_nodes, 1
        ut = sqrt((x(i,1)-x(j,1))**2 + (x(i,2)-x(j,2))**2) / sig
        u(i,j) = ut
        if (i==j) then
           K(i,j) = 1.0
           K_diff(i,j) = - c1_(ord,1)/(2*sig*sig)
           K_diff2(i,j) = c2_(ord-1,1)/(4*sig**4)
        else
           lpt = c_(ord+1, 1) + c_(ord+1,2)*ut + c_(ord+1,3)*ut**2 + c_(ord+1,4)*ut**3 + c_(ord+1,5)*ut**4
           !lpt = (105 + 105*ut + 45*ut**2 + 10*ut**3 + ut**4)/105
           K(i,j) = lpt * exp(-1.0*ut)
           lpt = c1_(ord,1) + c1_(ord,2)*ut + c1_(ord,3)*ut**2 + c1_(ord,4)*ut**3 
           !lpt = (15 + 15*ut + 6*ut**2 + ut**3)/105
           K_diff(i,j) = -lpt * exp(-1.0*ut)/(2*sig**2)
           lpt = c2_(ord-1,1) + c2_(ord-1,2)*ut + c2_(ord-1,3)*ut**2 
           !lpt = (3 + 3*ut + ut**2)/105
           K_diff2(i,j) = lpt * exp(-1.0*ut)/(4*sig**4)
        end if
     end do
  end do
  !$omp end parallel do
end subroutine kernelMatrixLaplacianPrecompute

subroutine shoot1order(dt,kvs,kvo,alpha,x0,&
     num_times,num_nodes,dim,x,v)
  implicit none
  integer :: num_nodes, num_times, dim, kvo
  real(8) :: dt
  real(8) :: kvs
  real(8) :: x(num_nodes, dim, num_times)
  real(8) :: x0(num_nodes, dim)
  real(8) :: alpha(num_nodes, dim, num_times)
  real(8) :: v(num_nodes, dim, num_times)

  real(8) :: kv_ut, Kv, Kv_diff, lpt
  real(8) :: ada
  real(8) :: x_diff(dim)
  integer :: t, k, l, ov
  real(8) :: dx(dim)
  real(8) :: c_(5, 5), c1_(4, 4)

  !f2py integer, intent(in) :: num_nodes, num_times, dim, kvo
  !f2py real(8), intent(in) :: dt, kvs
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: x0
  !f2py real(8), intent(in), dimension(num_nodes, dim, num_times) :: alpha
  !f2py real(8), intent(out), dimension(num_nodes, dim, num_times) :: x
  !f2py real(8), intent(out), dimension(num_nodes, dim, num_times) :: v

  x(:,:,1) = x0
  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
  c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
       0.,0.,0., 1./105 /), (/4,4/))

  do t = 1, num_times-1, 1
     !$omp parallel do private(k,kv_ut,lpt,Kv, &
     !$omp& x_diff,dx) shared (alpha,x,kvs,kvo,c_,c1_)
     do k = 1, num_nodes, 1
        dx=0
        do l=1, num_nodes, 1
           kv_ut = norm2(x(k,:,t)-x(l,:,t)) / kvs 
           if (kvo > 4) then
              kv_ut = kv_ut * kv_ut
              if (k==l) then
                 Kv = 1.0
              else
                 Kv = exp(-1.0*kv_ut)
              end if
           else           
              if (k==l) then
                 Kv = 1.0
              else
                 lpt = c_(kvo+1, 1) + c_(kvo+1,2)*kv_ut + c_(kvo+1,3)*kv_ut**2 + c_(kvo+1,4)*kv_ut**3 + c_(kvo+1,5)*kv_ut**4
                 Kv = lpt * exp(-1.0*kv_ut)
              end if
           end if

           dx = dx + Kv*alpha(l,:,t)
        end do !l
        x(k,:,t+1) = x(k,:,t) + dt*dx
        v(k,:,t) = dx
     end do !k
     !$omp end parallel do
  end do !t

end subroutine shoot1order

subroutine adjoint1order(dt,kvs,kvo,alpha,x0,&
     da,dx,num_times,num_nodes,dim,x,ex,ea)
  implicit none
  integer :: num_nodes, num_times, dim, kvo
  real(8) :: dt
  real(8) :: kvs
  real(8) :: x0(num_nodes, dim)
  real(8) :: alpha(num_nodes, dim, num_times)
  real(8) :: dx(num_nodes,dim)
  real(8) :: da(num_nodes,dim)
  real(8) :: x(num_nodes,dim,num_times)
  real(8) :: ex(num_nodes,dim,num_times)
  real(8) :: ea(num_nodes,dim,num_times)

  real(8) :: kv_ut, Kv, Kv_diff, lpt
  real(8) :: ada
  real(8) :: x_diff(dim)
  integer :: t, k, l, ov
  real(8) :: dex(num_nodes, dim)
  real(8) :: c_(5, 5), c1_(4, 4)
  real(8) :: v(num_nodes,dim,num_times)

  !f2py integer, intent(in) :: num_nodes, num_times, dim, kvo
  !f2py real(8), intent(in) :: dt, kvs
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: x0, dx, da
  !f2py real(8), intent(in), dimension(num_nodes, dim, num_times) :: alpha 
  !f2py real(8), intent(out), dimension(num_nodes, dim, num_times) :: x
  !f2py real(8), intent(out), dimension(num_nodes, dim, num_times) :: ex,ea 

  call shoot1order(dt,kvs,kvo,alpha,x0,num_times,num_nodes,dim,x,v)
  ex(:,:,num_times) = dx
  ea(:,:,num_times) = da
  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
  c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
       0.,0.,0., 1./105 /), (/4,4/))

  do t=num_times,2,-1
     !$omp parallel do private(k,kv_ut,lpt,Kv,Kv_diff, &
     !$omp& ada,x_diff) shared (dt,alpha,x,ex,ea, &
     !$omp& kvs,kvo,dex,c_,c1_)
     do k=1,num_nodes,1
        dex(k,:) = 0
        do l=1,num_nodes,1

           x_diff = x(k,:,t-1)-x(l,:,t-1)
           ada = dot_product(alpha(k,:,t-1),alpha(l,:,t-1))
           kv_ut = norm2(x_diff)  / kvs 

           if (kvo > 4) then
              kv_ut = kv_ut * kv_ut
              if (k==l) then
                 Kv = 1.0
                 Kv_diff = -1/(2*kvs*kvs)
              else
                 Kv = exp(-1.0*kv_ut)
                 Kv_ut = (-1.0)*exp(-1.0*kv_ut)/(2*kvs*kvs)
              end if
           else           
              if (k==l) then
                 Kv = 1.0
                 Kv_diff = - c1_(kvo,1)/(2*kvs*kvs)
              else
                 lpt = c_(kvo+1, 1) + c_(kvo+1,2)*kv_ut + c_(kvo+1,3)*kv_ut**2 + c_(kvo+1,4)*kv_ut**3 + c_(kvo+1,5)*kv_ut**4
                 Kv = lpt * exp(-1.0*kv_ut)
                 lpt = c1_(kvo,1) + c1_(kvo,2)*kv_ut + c1_(kvo,3)*kv_ut**2 + c1_(kvo,4)*kv_ut**3 
                 Kv_diff = -lpt * exp(-1.0*kv_ut)/(2*kvs**2)
              end if
           end if

           dex(k,:) = dex(k,:) + (-1)*( &
                Kv_diff*2.0*x_diff*dot_product(alpha(l,:,t-1),ex(k,:,t)) &
                + Kv_diff*2.0*x_diff*dot_product(alpha(k,:,t-1),ex(l,:,t)) ) &
                + (2)*(Kv_diff*2.0*x_diff*dot_product(alpha(l,:,t-1),alpha(k,:,t))) 

        end do ! l
	ex(k,:,t-1) = ex(k,:,t) - dt*dex(k,:)
     end do ! k
     !$omp end parallel do
  end do ! t
end subroutine adjoint1order


subroutine shoot2order(dt,kvs,kvo,alpha0,x0,&
     num_times,num_nodes,dim,x,alpha,v)
  implicit none
  integer :: num_nodes, num_times, dim, kvo
  real(8) :: dt
  real(8) :: kvs
  real(8) :: x(num_nodes, dim, num_times)
  real(8) :: x0(num_nodes, dim)
  real(8) :: alpha0(num_nodes, dim)
  real(8) :: alpha(num_nodes, dim, num_times)
  real(8) :: v(num_nodes, dim, num_times)

  real(8) :: kv_ut, Kv, Kv_diff, lpt
  real(8) :: ada
  real(8) :: x_diff(dim)
  integer :: t, k, l, ov
  real(8) :: dx(dim),da(dim)
  real(8) :: c_(5, 5), c1_(4, 4)

  !f2py integer, intent(in) :: num_nodes, num_times, dim, kvo
  !f2py real(8), intent(in) :: dt, kvs
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: x0
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: alpha0 
  !f2py real(8), intent(out), dimension(num_nodes, dim, num_times) :: x
  !f2py real(8), intent(out), dimension(num_nodes, dim, num_times) :: alpha
  !f2py real(8), intent(out), dimension(num_nodes, dim, num_times) :: v

  x(:,:,1) = x0
  alpha(:,:,1) = alpha0
  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
  c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
       0.,0.,0., 1./105 /), (/4,4/))

  do t = 1, num_times-1, 1
     !$omp parallel do private(k,kv_ut,lpt,Kv,Kv_diff, &
     !$omp& ada,x_diff,dx,da) shared (alpha,x,kvs,kvo,c_,c1_)
     do k = 1, num_nodes, 1
        dx=0
        da=0

        do l=1, num_nodes, 1
           kv_ut = norm2(x(k,:,t)-x(l,:,t)) 
           
           if (kvo > 4) then
              kv_ut = kv_ut * kv_ut
              if (k==l) then
                 Kv = 1.0
                 Kv_diff = - 1.0/(2*kvs*kvs)
              else
                 Kv = exp(-1.0*kv_ut)
                 Kv_ut = -1.0*exp(-1.0*kv_ut)/(2*kvs*kvs)
              end if
           else           
              if (k==l) then
                 Kv = 1.0
                 Kv_diff = - c1_(kvo,1)/(2*kvs*kvs)
              else
                 lpt = c_(kvo+1, 1) + c_(kvo+1,2)*kv_ut + c_(kvo+1,3)*kv_ut**2 + c_(kvo+1,4)*kv_ut**3 + c_(kvo+1,5)*kv_ut**4
                 Kv = lpt * exp(-1.0*kv_ut)
                 lpt = c1_(kvo,1) + c1_(kvo,2)*kv_ut + c1_(kvo,3)*kv_ut**2 + c1_(kvo,4)*kv_ut**3 
                 Kv_diff = -lpt * exp(-1.0*kv_ut)/(2*kvs**2)
              end if
           end if

           dx = dx + Kv*alpha(l,:,t)
           ada = dot_product(alpha(l,:,t), alpha(k,:,t))
           x_diff = x(k,:,t)-x(l,:,t)
           da = da - Kv_diff*ada*2* x_diff
        end do !l
        x(k,:,t+1) = x(k,:,t) + dt*dx
        v(k,:,t) = dx
        alpha(k,:,t+1) = alpha(k,:,t) + dt*da
     end do !k
     !$omp end parallel do
  end do !t

end subroutine shoot2order



subroutine computeHamiltonian_u(sfactor,kvs,kvo,khs,kho, alpha, x, z, ex, ez, em, num_nodes, H)
  integer :: num_nodes, kvo, kho
  real(8) :: sfactor,H
  real(8) :: kvs, khs
  real(8) :: x(num_nodes, 3)
  real(8) :: z(num_nodes, 3)
  real(8) :: alpha(num_nodes)
  real(8) :: ex(num_nodes,3)
  real(8) :: ez(num_nodes,3)
  real(8) :: em(num_nodes)

  real(8) :: kv_ut, kh_ut, Kv, Kv_diff, Kh, Kh_diff, lpt
  real(8) :: Kv_diff2, Kh_diff2
  real(8) :: x_diff(3), zdz
  integer :: t, k, l
  real(8) :: dex(3), dez(3), dea
  real(8) :: c_(5, 5), c1_(4, 4)

  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
  c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
       0.,0.,0., 1./105 /), (/4,4/))
  H = 0
  do k = 1, num_nodes, 1
     dx=0
     dz=0
     dm=0

     do l=1, num_nodes, 1

        kv_ut = norm2(x(k,:)-x(l,:)) / kvs 
        kh_ut = norm2(x(k,:)-x(l,:)) / khs 

        if (k==l) then
           Kv = 1.0
           Kv_diff = - c1_(kvo,1)/(2*kvs*kvs)
           Kh = 1.0
           Kh_diff = - c1_(kho,1)/(2*khs*khs)
        else
           lpt = c_(kvo+1, 1) + c_(kvo+1,2)*kv_ut + c_(kvo+1,3)*kv_ut**2 + c_(kvo+1,4)*kv_ut**3 + c_(kvo+1,5)*kv_ut**4
           Kv = lpt * exp(-1.0*kv_ut)
           lpt = c_(kho+1, 1) + c_(kho+1,2)*kh_ut + c_(kho+1,3)*kh_ut**2 + c_(kho+1,4)*kh_ut**3 + c_(kho+1,5)*kh_ut**4
           Kh = lpt * exp(-1.0*kh_ut)
           lpt = c1_(kvo,1) + c1_(kvo,2)*kv_ut + c1_(kvo,3)*kv_ut**2 + c1_(kvo,4)*kv_ut**3 
           Kv_diff = -lpt * exp(-1.0*kv_ut)/(2*kvs**2)
           lpt = c1_(kho,1) + c1_(kho,2)*kh_ut + c1_(kho,3)*kh_ut**2 + c1_(kho,4)*kh_ut**3 
           Kh_diff = -lpt * exp(-1.0*kh_ut)/(2*khs**2)
        end if

        H = H + sfactor*Kv*dot_product(ex(k, :), z(l,:))
        zdz = dot_product(z(k,:),z(l,:))
        x_diff = x(k,:)-x(l,:)
        H = H - sfactor*Kv_diff*zdz*2* &
             dot_product(ez(k, :), x_diff) - Kh_diff*alpha(k)*alpha(l)*2*dot_product(ez(k, :), x_diff)
        H = H + Kh*alpha(l)*em(k)

     end do !l
  end do !k

end subroutine computeHamiltonian_u

subroutine testHamiltonian_u(sfactor,kvs,kvo,khs,kho, alpha, x, z, ex,&
     ez, em, dex, dez, dea, num_nodes)
  integer :: num_nodes, kvo, kho
  real(8) :: sfactor,H
  real(8) :: kvs, khs
  real(8) :: x(num_nodes, 3)
  real(8) :: z(num_nodes, 3)
  real(8) :: alpha(num_nodes)
  real(8) :: ex(num_nodes,3)
  real(8) :: ez(num_nodes,3)
  real(8) :: em(num_nodes)
  real(8) :: dex(num_nodes,3)
  real(8) :: dez(num_nodes,3)
  real(8) :: dea(num_nodes)

  real(8) :: H0, H1, eps, test
  real(8) :: vx(num_nodes, 3)
  real(8) :: vz(num_nodes, 3)
  real(8) :: va(num_nodes)

  eps = 1e-8
  call computeHamiltonian_u(sfactor,kvs,kvo,khs,kho, alpha, &
       x, z, ex, ez, em, num_nodes, H0)
  call random_number(vx)
  vx = vx-0.5
  call computeHamiltonian_u(sfactor,kvs,kvo,khs,kho, &
       alpha, x+eps*vx, z, ex, ez, em, num_nodes, H1)
  test = dot_product(dex(:,1), vx(:,1)) + &
       dot_product(dex(:,2), vx(:,2)) + dot_product(dex(:,3), vx(:,3))
  print *, 'Test x', (H1-H0)/eps, -test

  call random_number(vz)
  vz = vz-0.5
  call computeHamiltonian_u(sfactor,kvs,kvo,khs,kho, &
       alpha, x, z+eps*vz, ex, ez, em, num_nodes, H1)
  test = dot_product(dez(:,1), vz(:,1)) + &
       dot_product(dez(:,2), vz(:,2)) + dot_product(dez(:,3), vz(:,3))
  print *, 'Test z', (H1-H0)/eps, -test
  call random_number(va)
  va = va-0.5
  call computeHamiltonian_u(sfactor,kvs,kvo,khs,kho, &
       alpha+eps*va, x, z, ex, ez, em, num_nodes, H1)
  test = dot_product(dea, va) 
  print *, 'Test a', (H1-H0)/eps, -test

end subroutine testHamiltonian_u



subroutine adjoint2order(dt,kvs,kvo,alpha,x,&
     da,dx,num_times,num_nodes,dim,ea,ex)
  implicit none
  integer :: num_nodes, num_times, dim, kvo
  real(8) :: dt
  real(8) :: kvs
  real(8) :: x(num_nodes, dim, num_times)
  real(8) :: alpha(num_nodes,dim,num_times)
  real(8) :: dx(num_nodes,dim)
  real(8) :: da(num_nodes,dim)
  real(8) :: ex(num_nodes,dim,num_times)
  real(8) :: ea(num_nodes,dim,num_times)

  real(8) :: kv_ut, Kv, Kv_diff, lpt
  real(8) :: Kv_diff2
  real(8) :: x_diff(dim), ada
  integer :: t, k, l
  real(8) :: dex(num_nodes, dim), dea(num_nodes, dim)
  real(8) :: c_(5, 5), c1_(4, 4), c2_(3,3)

  !f2py integer, intent(in) :: num_nodes, num_times, dim, kvo
  !f2py real(8), intent(in) :: dt, kvs
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: dx,da 
  !f2py real(8), intent(in), dimension(num_nodes, dim, num_times) :: alpha 
  !f2py real(8), intent(in), dimension(num_nodes, dim, num_times) :: x
  !f2py real(8), intent(out), dimension(num_nodes, dim, num_times) :: ex 
  !f2py real(8), intent(out), dimension(num_nodes, dim, num_times) :: ea 

  ex(:,:,num_times) = dx
  ea(:,:,num_times) = da

  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
  c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
       0.,0.,0., 1./105 /), (/4,4/))
  c2_ = reshape((/ 1.0/3, 1./15, 1./35,    0., 1./15, 1./35,   0.,0., 1./105 /), (/3,3/))
  !print *, c_(kho+1,:)

  do t=num_times,2,-1
     !$omp parallel do private(k,l,kv_ut,lpt,Kv,Kv_diff, &
     !$omp& Kv_diff2, & 
     !$omp& ada,x_diff) shared (dt,alpha,x,ex,ea, &
     !$omp& kvs,kvo,dex,dea,c_,c1_,c2_)
     do k=1,num_nodes,1
        dex(k,:) = 0
        dea(k,:) = 0
        do l=1,num_nodes,1

           x_diff = x(k,:,t-1)-x(l,:,t-1)
           ada = dot_product(alpha(k,:,t-1),alpha(l,:,t-1))
           kv_ut = norm2(x_diff) / kvs 

           if (kvo > 4) then
              kv_ut = kv_ut * kv_ut
              if (k==l) then
                 Kv = 1.0
                 Kv_diff = -1.0/(2*kvs*kvs)
                 Kv_diff2 = 1/(4*kvs**4)
              else
                 Kv = exp(-1.0*kv_ut)
                 Kv_diff = -1.0* exp(-1.0*kv_ut)/(2*kvs*kvs)
                 Kv_diff2 = exp(-1.0*kv_ut)/(4*kvs**4)
              end if
           else           
              if (k==l) then
                 Kv = 1.0
                 Kv_diff = - c1_(kvo,1)/(2*kvs*kvs)
                 Kv_diff2 = c2_(kvo-1,1)/(4*kvs**4)
              else
                 lpt = c_(kvo+1, 1) + c_(kvo+1,2)*kv_ut + c_(kvo+1,3)*kv_ut**2 + c_(kvo+1,4)*kv_ut**3 + c_(kvo+1,5)*kv_ut**4
                 Kv = lpt * exp(-1.0*kv_ut)
                 lpt = c1_(kvo,1) + c1_(kvo,2)*kv_ut + c1_(kvo,3)*kv_ut**2 + c1_(kvo,4)*kv_ut**3 
                 Kv_diff = -lpt * exp(-1.0*kv_ut)/(2*kvs**2)
                 lpt = c2_(kvo-1,1) + c2_(kvo-1,2)*kv_ut + c2_(kvo-1,3)*kv_ut**2 
                 Kv_diff2 = lpt * exp(-1.0*kv_ut)/(4*kvs**4)
              end if
           end if

           dex(k,:) = dex(k,:) + (-1)*( &
                Kv_diff*2.0*x_diff*dot_product(alpha(l,:,t-1),ex(k,:,t)) &
                + Kv_diff*2.0*x_diff*dot_product(alpha(k,:,t-1),ex(l,:,t)) ) &
                + ( ada*(x_diff*Kv_diff2*4*dot_product(x_diff,ea(k,:,t)) &
                +2*Kv_diff*ea(k,:,t)) &
                + ada*(-1*x_diff*Kv_diff2*4*dot_product(x_diff,ea(l,:,t)) &
                -2*Kv_diff*ea(l,:,t)) )

           dea(k,:) = dea(k,:) + (-1)*Kv*ex(l,:,t) & 
                + alpha(l,:,t-1)*Kv_diff*2*dot_product(x_diff,ea(k,:,t)) &
                + alpha(l,:,t-1)*Kv_diff*2*dot_product(-1*x_diff,ea(l,:,t)) 
        end do ! l
	ex(k,:,t-1) = ex(k,:,t) - dt*dex(k,:)
	ea(k,:,t-1) = ea(k,:,t) - dt*dea(k,:)
     end do ! k
     ! call testHamiltonian_u(sfactor,kvs,kvo,khs,kho, alpha, x(:,:,t-1), z(:,:,t-1), &
     !      ex(:,:,t), ez(:,:,t), em(:,t), dex, &
     !      dez, dea, num_nodes)
     !$omp end parallel do
  end do ! t
end subroutine adjoint2order

