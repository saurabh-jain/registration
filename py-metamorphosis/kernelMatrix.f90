
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

subroutine shoot(dt,sfactor,kvs,kvo,khs,kho,alpha,x0,m0,z0,&
     num_times,num_nodes,x,m,z,J,v)
  implicit none
  integer :: num_nodes, num_times, kvo, kho
  real(8) :: sfactor, dt
  real(8) :: kvs, khs
  real(8) :: x(num_nodes, 3, num_times)
  real(8) :: x0(num_nodes, 3)
  real(8) :: m(num_nodes, num_times)
  real(8) :: m0(num_nodes)
  real(8) :: z(num_nodes, 3, num_times)
  real(8) :: z0(num_nodes, 3)
  real(8) :: J(num_nodes, num_times)
  real(8) :: alpha(num_nodes)
  real(8) :: v(num_nodes, 3, num_times)

  real(8) :: kv_ut, kh_ut, Kv, Kv_diff, Kh, Kh_diff, lpt
  real(8) :: zdz, kvz
  real(8) :: x_diff(3)
  integer :: t, k, l, ov, oh
  real(8) :: dx(3), dz(3), dm, dJ
  real(8) :: c_(5, 5), c1_(4, 4)

  !f2py integer, intent(in) :: num_nodes, num_times, kvo, kho
  !f2py real(8), intent(in) :: sfactor, dt, kvs, khs
  !f2py real(8), intent(in), dimension(num_nodes, 3) :: x0
  !f2py real(8), intent(in), dimension(num_nodes) :: alpha 
  !f2py real(8), intent(in), dimension(num_nodes) :: m0
  !f2py real(8), intent(in), dimension(num_nodes, 3):: z0
  !f2py real(8), intent(out), dimension(num_nodes, 3, num_times) :: x
  !f2py real(8), intent(out), dimension(num_nodes, num_times) :: J
  !f2py real(8), intent(out), dimension(num_nodes, num_times) :: m
  !f2py real(8), intent(out), dimension(num_nodes, 3, num_times) :: z
  !f2py real(8), intent(out), dimension(num_nodes, 3, num_times) :: v

  x(:,:,1) = x0
  m(:,1) = m0
  z(:,:,1) = z0
  J(:,1) = 1.0
  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
  c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
       0.,0.,0., 1./105 /), (/4,4/))

  do t = 1, num_times-1, 1
     !$omp parallel do private(k,kv_ut,kh_ut,lpt,Kv,Kv_diff,Kh,Kh_diff, &
     !$omp& zdz,x_diff,kvz,dx,dz,dm,dJ) shared (alpha,x,m,z,J,kvs,khs,kvo,kho,c_,c1_)
     do k = 1, num_nodes, 1
        dx=0
        dz=0
        dm=0
        dJ=0

        do l=1, num_nodes, 1
           kv_ut = sqrt((x(k,1,t)-x(l,1,t))**2 + (x(k,2,t)-x(l,2,t))**2) / kvs 
           kh_ut = sqrt((x(k,1,t)-x(l,1,t))**2 + (x(k,2,t)-x(l,2,t))**2) / khs 
           
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

           dx = dx + sfactor*Kv*alpha(l)*z(l,:,t)
           zdz = z(l,1,t)*z(k,1,t) + z(l,2,t)*z(k,2,t) + z(l,3,t)*z(k,3,t)
           x_diff = x(k,:,t)-x(l,:,t)
           dz = dz - sfactor*Kv_diff*alpha(l)*zdz*2* &
                x_diff - Kh_diff*alpha(l)*2*x_diff
           dm = dm + 0*sfactor*Kv*alpha(l)*zdz + Kh*alpha(l)
           kvz = dot_product(x_diff,z(l,:,t))
           dJ = dJ + sfactor*J(k,t)*alpha(l)*Kv_diff*2*kvz
        end do !l
        x(k,:,t+1) = x(k,:,t) + dt*dx
        v(k,:,t) = dx
        m(k,t+1) = m(k,t) + dt*dm
        z(k,:,t+1) = z(k,:,t) + dt*dz
        J(k,t+1) = J(k,t) + dt*dJ
     end do !k
     !$omp end parallel do

  end do !t

end subroutine shoot

subroutine shoot_NoJ(dt,sfactor,kvs,kvo,khs,kho,alpha,x0,m0,z0,&
     num_times,num_nodes,x,m,z,v)
  implicit none
  integer :: num_nodes, num_times, kvo, kho
  real(8) :: sfactor, dt
  real(8) :: kvs, khs
  real(8) :: x(num_nodes, 3, num_times)
  real(8) :: x0(num_nodes, 3)
  real(8) :: m(num_nodes, num_times)
  real(8) :: m0(num_nodes)
  real(8) :: z(num_nodes, 3, num_times)
  real(8) :: z0(num_nodes, 3)
  real(8) :: alpha(num_nodes)
  real(8) :: v(num_nodes, 3, num_times)

  real(8) :: kv_ut, kh_ut, Kv, Kv_diff, Kh, Kh_diff, lpt
  real(8) :: zdz
  real(8) :: x_diff(3)
  integer :: t, k, l, ov, oh
  real(8) :: dx(3), dz(3), dm
  real(8) :: c_(5, 5), c1_(4, 4), c2_(3,3)

  !f2py integer, intent(in) :: num_nodes, num_times, kvo, kho
  !f2py real(8), intent(in) :: sfactor, dt, kvs, khs
  !f2py real(8), intent(in), dimension(num_nodes, 3) :: x0
  !f2py real(8), intent(in), dimension(num_nodes) :: alpha 
  !f2py real(8), intent(in), dimension(num_nodes) :: m0
  !f2py real(8), intent(in), dimension(num_nodes, 3):: z0
  !f2py real(8), intent(out), dimension(num_nodes, 3, num_times) :: x
  !f2py real(8), intent(out), dimension(num_nodes, num_times) :: m
  !f2py real(8), intent(out), dimension(num_nodes, 3, num_times) :: z
  !f2py real(8), intent(out), dimension(num_nodes, 3, num_times) :: v

  x(:,:,1) = x0
  m(:,1) = m0
  z(:,:,1) = z0
  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
  c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
       0.,0.,0., 1./105 /), (/4,4/))

  do t = 1, num_times-1, 1
     !$omp parallel do private(k,kv_ut,kh_ut,lpt,Kv,Kv_diff,Kh,Kh_diff, &
     !$omp& zdz,x_diff,dx,dz,dm) shared (alpha,x,m,z,kvs,khs,kvo,kho,c_,c1_)
     do k = 1, num_nodes, 1
        dx=0
        dz=0
        dm=0

        do l=1, num_nodes, 1

           kv_ut = norm2(x(k,:,t)-x(l,:,t)) / kvs 
           kh_ut = norm2(x(k,:,t)-x(l,:,t)) / khs 

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


           dx = dx + sfactor*Kv*alpha(l)*z(l,:,t)
           zdz = dot_product(z(k,:,t),z(l,:,t))
           x_diff = x(k,:,t)-x(l,:,t)
           dz = dz - sfactor*Kv_diff*alpha(l)*zdz*2* &
                x_diff - Kh_diff*alpha(l)*2*x_diff
           dm = dm + 0*sfactor*Kv*alpha(l)*zdz + Kh*alpha(l)

        end do !l
        x(k,:,t+1) = x(k,:,t) + dt*dx
        v(k,:,t) = dx
        m(k,t+1) = m(k,t) + dt*dm
        z(k,:,t+1) = z(k,:,t) + dt*dz
     end do !k
     !$omp end parallel do

  end do !t

end subroutine shoot_NoJ




subroutine shoot_unconstrained(dt,sfactor,kvs,kvo,khs,kho,alpha,x0,m0,z0,&
     num_times,num_nodes,x,m,z,v)
  implicit none
  integer :: num_nodes, num_times, kvo, kho
  real(8) :: sfactor, dt
  real(8) :: kvs, khs
  real(8) :: x(num_nodes, 3, num_times)
  real(8) :: x0(num_nodes, 3)
  real(8) :: m(num_nodes, num_times)
  real(8) :: m0(num_nodes)
  real(8) :: z(num_nodes, 3, num_times)
  real(8) :: z0(num_nodes, 3)
  real(8) :: alpha(num_nodes)
  real(8) :: v(num_nodes, 3, num_times)

  real(8) :: kv_ut, kh_ut, Kv, Kv_diff, Kh, Kh_diff, lpt
  real(8) :: zdz
  real(8) :: x_diff(3)
  integer :: t, k, l
  real(8) :: dx(3), dz(3), dm
  real(8) :: c_(5, 5), c1_(4, 4)

  !f2py integer, intent(in) :: num_nodes, num_times, kvo, kho
  !f2py real(8), intent(in) :: sfactor, dt, kvs, khs
  !f2py real(8), intent(in), dimension(num_nodes, 3) :: x0
  !f2py real(8), intent(in), dimension(num_nodes) :: alpha 
  !f2py real(8), intent(in), dimension(num_nodes) :: m0
  !f2py real(8), intent(in), dimension(num_nodes, 3):: z0
  !f2py real(8), intent(out), dimension(num_nodes, 3, num_times) :: x
  !f2py real(8), intent(out), dimension(num_nodes, num_times) :: m
  !f2py real(8), intent(out), dimension(num_nodes, 3, num_times) :: z
  !f2py real(8), intent(out), dimension(num_nodes, 3, num_times) :: v

  x(:,:,1) = x0
  m(:,1) = m0
  z(:,:,1) = z0
  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
  c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
       0.,0.,0., 1./105 /), (/4,4/))

  do t = 1, num_times-1, 1
     !$omp parallel do private(k,kv_ut,kh_ut,lpt,Kv,Kv_diff,Kh,Kh_diff, &
     !$omp& zdz,x_diff,dx,dz,dm) shared (alpha,x,m,z,kvs,khs,kvo,kho,c_,c1_)
     do k = 1, num_nodes, 1
        dx=0
        dz=0
        dm=0

        do l=1, num_nodes, 1

           kv_ut = norm2(x(k,:,t)-x(l,:,t)) / kvs 
           kh_ut = norm2(x(k,:,t)-x(l,:,t)) / khs 

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

           dx = dx + sfactor*Kv*z(l,:,t)
           zdz = dot_product(z(k,:,t),z(l,:,t))
           x_diff = x(k,:,t)-x(l,:,t)
           dz = dz - sfactor*Kv_diff*zdz*2* &
                x_diff - Kh_diff*alpha(k)*alpha(l)*2*x_diff
           dm = dm + Kh*alpha(l)

	end do !l
	x(k,:,t+1) = x(k,:,t) + dt*dx
	v(k,:,t) = dx
	m(k,t+1) = m(k,t) + dt*dm
	z(k,:,t+1) = z(k,:,t) + dt*dz
     end do !k
     !$omp end parallel do

  end do !t

end subroutine shoot_unconstrained

subroutine adjointSystem(dt,sfactor,kvs,kvo,khs,kho,alpha,x,m,z,J,&
     dx,dm,dJ,num_times,num_nodes,ealpha,ex)
  implicit none
  integer :: num_nodes, num_times, kvo, kho
  real(8) :: sfactor, dt
  real(8) :: kvs, khs
  real(8) :: x(num_nodes, 3, num_times)
  real(8) :: m(num_nodes, num_times)
  real(8) :: z(num_nodes, 3, num_times)
  real(8) :: J(num_nodes, num_times)
  real(8) :: alpha(num_nodes)
  real(8) :: dx(num_nodes,3)
  real(8) :: dm(num_nodes)
  real(8) :: dJ(num_nodes)
  real(8) :: ex(num_nodes,3,num_times)
  real(8) :: ez(num_nodes,3,num_times)
  real(8) :: em(num_nodes,num_times)
  real(8) :: eJ(num_nodes,num_times)
  real(8) :: ealpha(num_nodes)

  real(8) :: kv_ut, kh_ut, Kv, Kv_diff, Kh, Kh_diff, lpt
  real(8) :: Kv_diff2, Kh_diff2
  real(8) :: x_diff(3), zdz
  integer :: t, k, l
  real(8) :: dex(3), dez(3), deJ, dea
  real(8) :: c_(5, 5), c1_(4, 4), c2_(3,3)

  !f2py integer, intent(in) :: num_nodes, num_times, kvo, kho
  !f2py real(8), intent(in) :: sfactor, dt, kvs, khs
  !f2py real(8), intent(in), dimension(num_nodes) :: alpha 
  !f2py real(8), intent(in), dimension(num_nodes, 3, num_times) :: x
  !f2py real(8), intent(in), dimension(num_nodes, num_times) :: J
  !f2py real(8), intent(in), dimension(num_nodes, num_times) :: m
  !f2py real(8), intent(in), dimension(num_nodes, 3, num_times) :: z
  !f2py real(8), intent(out), dimension(num_nodes) :: ealpha 
  !f2py real(8), intent(out), dimension(num_nodes, 3, num_times) :: ex 

  ex(:,:,num_times) = dx
  em(:,num_times) = dm
  eJ(:,num_times) = dJ
  ealpha = 0
  ez(:,:,num_times) = 0
  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
  c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
       0.,0.,0., 1./105 /), (/4,4/))
  c2_ = reshape((/ 1.0/3, 1./15, 1./35,    0., 1./15, 1./35,   0.,0., 1./105 /), (/3,3/))

  do t=num_times,2,-1
     !$omp parallel do private(k,kv_ut,kh_ut,lpt,Kv,Kv_diff,Kh,Kh_diff, &
     !$omp& Kv_diff2,Kh_diff2,dex,dez,dea,deJ, & 
     !$omp& zdz,x_diff) shared (dt,alpha,x,m,z,J,ex,ez,em,eJ,ealpha, &
     !$omp& kvs,khs,kvo,kho,c_,c1_,c2_)
     do k=1,num_nodes,1
        dex = 0
        dez = 0
        dea = 0
        deJ = 0
        do l=1,num_nodes,1

           !kv_ut = sqrt((x(k,1,t)-x(l,1,t))**2 + (x(k,2,t)-x(l,2,t))**2) / kvs 
           !kh_ut = sqrt((x(k,1,t)-x(l,1,t))**2 + (x(k,2,t)-x(l,2,t))**2) / khs 
           kv_ut = norm2(x(k,:,t-1)-x(l,:,t-1)) / kvs 
           kh_ut = norm2(x(k,:,t-1)-x(l,:,t-1)) / khs 

           if (k==l) then
              Kv = 1.0
              Kv_diff = - c1_(kvo,1)/(2*kvs*kvs)
              Kh = 1.0
              Kh_diff = - c1_(kho,1)/(2*khs*khs)
              Kv_diff2 = c2_(kvo-1,1)/(4*kvs**4)
              Kh_diff2 = c2_(kho-1,1)/(4*khs**4)
           else
              lpt = c_(kvo+1, 1) + c_(kvo+1,2)*kv_ut + c_(kvo+1,3)*kv_ut**2 + c_(kvo+1,4)*kv_ut**3 + c_(kvo+1,5)*kv_ut**4
              Kv = lpt * exp(-1.0*kv_ut)
              lpt = c_(kho+1, 1) + c_(kho+1,2)*kh_ut + c_(kho+1,3)*kh_ut**2 + c_(kho+1,4)*kh_ut**3 + c_(kho+1,5)*kh_ut**4
              Kh = lpt * exp(-1.0*kh_ut)
              lpt = c1_(kvo,1) + c1_(kvo,2)*kv_ut + c1_(kvo,3)*kv_ut**2 + c1_(kvo,4)*kv_ut**3 
              Kv_diff = -lpt * exp(-1.0*kv_ut)/(2*kvs**2)
              lpt = c1_(kho,1) + c1_(kho,2)*kh_ut + c1_(kho,3)*kh_ut**2 + c1_(kho,4)*kh_ut**3 
              Kh_diff = -lpt * exp(-1.0*kh_ut)/(2*khs**2)
              lpt = c2_(kvo-1,1) + c2_(kvo-1,2)*kv_ut + c2_(kvo-1,3)*kv_ut**2 
              Kv_diff2 = lpt * exp(-1.0*kv_ut)/(4*kvs**4)
              lpt = c2_(kho-1,1) + c2_(kho-1,2)*kh_ut + c2_(kho-1,3)*kh_ut**2 
              Kh_diff2 = lpt * exp(-1.0*kh_ut)/(4*khs**4)
           end if

           x_diff = x(k,:,t-1)-x(l,:,t-1)
           zdz = dot_product(z(k,:,t-1),z(l,:,t-1))
           dex = dex + (-sfactor)*( &
                Kv_diff*2.0*x_diff*alpha(l)*dot_product(z(l,:,t-1),ex(k,:,t)) &
                + Kv_diff*2.0*x_diff*alpha(k)*dot_product(z(k,:,t-1),ex(l,:,t)) ) &
                + sfactor*( &
                alpha(l)*zdz*(x_diff*Kv_diff2*4*dot_product(x_diff,ez(k,:,t)) &
                +2*Kv_diff*ez(k,:,t)) &
                + alpha(k)*zdz*(-1*x_diff*Kv_diff2*4*dot_product(x_diff,ez(l,:,t)) &
                -2*Kv_diff*ez(l,:,t)) ) &
                + alpha(l)*(x_diff*Kh_diff2*4*dot_product(x_diff,ez(k,:,t)) &
                +2*Kh_diff*ez(k,:,t)) &
                + alpha(k)*(-1*x_diff*Kh_diff2*4*dot_product(x_diff,ez(l,:,t)) &
                -2*Kh_diff*ez(l,:,t)) & 
                - alpha(l)*Kh_diff*2*x_diff*em(k,t) &
                - alpha(k)*Kh_diff*2*x_diff*em(l,t) &
                - sfactor*( J(k,t)*alpha(l)*(x_diff*Kv_diff2*4*dot_product(x_diff,z(l,:,t-1)) &
                +2*Kv_diff*z(l,:,t-1))*eJ(k,t) &
                + J(l,t-1)*alpha(k)*(-1*x_diff*Kv_diff2*4*dot_product(x_diff,z(k,:,t-1)) &
                -2*Kv_diff*z(k,:,t-1))*eJ(l,t) )

           dez = dez + (-sfactor)*Kv*alpha(k)*ex(l,:,t) & 
                + sfactor*alpha(l)*z(l,:,t-1)*Kv_diff*2*dot_product(x_diff,ez(k,:,t)) &
                + sfactor*alpha(k)*z(l,:,t-1)*Kv_diff*2*dot_product(-1*x_diff,ez(l,:,t)) &
                - sfactor*Kv_diff*2*(-1)*x_diff*alpha(k)*J(l,t-1)*eJ(l,t)

           dea = dea + (-sfactor)*dot_product(z(k,:,t-1),Kv*ex(l,:,t)) & 
                + sfactor*dot_product(z(k,:,t-1),z(l,:,t-1))*Kv_diff*2* &
                dot_product(-1*x_diff,ez(l,:,t)) &
                + Kh_diff*2*dot_product(-1*x_diff,ez(l,:,t)) &
                - Kh*em(l,t-1) &
                - sfactor*J(l,t-1)*dot_product(z(k,:,t-1),-1*x_diff)*2*Kv_diff*eJ(l,t)

           deJ = deJ + (-sfactor)*alpha(l)*dot_product(z(l,:,t-1),x_diff)*2* &
                Kv_diff*eJ(k,t)
        end do ! l
	ex(k,:,t-1) = ex(k,:,t) - dt*dex
	ez(k,:,t-1) = ez(k,:,t) - dt*dez
	ealpha(k) = ealpha(k) - dt*dea
	eJ(k,t-1) = eJ(k,t) - dt*deJ
	em(k,t-1) = em(k,t)
     end do ! k
     !$omp end parallel do
  end do ! t
  ! t=1
  ! !$omp parallel do private(k,kv_ut,kh_ut,lpt,Kv,Kv_diff,Kh,Kh_diff, &
  ! !$omp& Kv_diff2,Kh_diff2,dex,dez,dea,deJ, & 
  ! !$omp& zdz,x_diff) shared (dt,alpha,x,m,z,J,ex,ez,em,eJ,ealpha, &
  ! !$omp& kvs,khs,kvo,kho,c_,c1_,c2_)
  ! do k=1,num_nodes,1
  !    dex = 0
  !    dez = 0
  !    dea = 0
  !    deJ = 0
  !    do l=1,num_nodes,1

  !       !kv_ut = sqrt((x(k,1,t)-x(l,1,t))**2 + (x(k,2,t)-x(l,2,t))**2) / kvs 
  !       !kh_ut = sqrt((x(k,1,t)-x(l,1,t))**2 + (x(k,2,t)-x(l,2,t))**2) / khs 
  !       kv_ut = norm2(x(k,:,t)-x(l,:,t)) / kvs 
  !       kh_ut = norm2(x(k,:,t)-x(l,:,t)) / khs 

  !       if (k==l) then
  !          Kv = 1.0
  !          Kv_diff = -1.0/((2*kvo-1)*2*kvs*kvs)
  !          Kh = 1.0
  !          Kh_diff = -1.0/((2*kho-1)*2*khs*khs)
  !       else
  !          lpt = (105 + 105*kv_ut + 45*kv_ut**2 + 10*kv_ut**3 + kv_ut**4)/105
  !          Kv = lpt * exp(-1.0*kv_ut)
  !          lpt = (105 + 105*kh_ut + 45*kh_ut**2 + 10*kh_ut**3 + kh_ut**4)/105
  !          Kh = lpt * exp(-1.0*kh_ut)
  !          lpt = (15 + 15*kv_ut + 6*kv_ut**2 + kv_ut**3)/105
  !          Kv_diff = -lpt * exp(-1.0*kv_ut)/(2*kvs**2)
  !          lpt = (15 + 15*kh_ut + 6*kh_ut**2 + kh_ut**3)/105
  !          Kh_diff = -lpt * exp(-1.0*kh_ut)/(2*khs**2)
  !       end if

  !       x_diff = x(k,:,t)-x(l,:,t)

  !       dea = dea + (-sfactor)*dot_product(z(k,:,t),Kv*ex(l,:,t)) & 
  !            + sfactor*dot_product(z(k,:,t),z(l,:,t))*Kv_diff*2* &
  !            dot_product(-1*x_diff,ez(l,:,t)) &
  !            + Kh_diff*2*dot_product(-1*x_diff,ez(l,:,t)) &
  !            - Kh*em(l,t) &
  !            - sfactor*J(l,t)*dot_product(z(k,:,t),-1*x_diff)*2*Kv_diff*eJ(l,t)

  !    end do ! l
  !    ealpha(k) = ealpha(k) - dt*dea
  ! end do ! k
  ! !$omp end parallel do

end subroutine adjointSystem

subroutine computeHamiltonian(sfactor,kvs,kvo,khs,kho, alpha, x, z, ex, ez, em, num_nodes, H)
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

  H = 0
  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
  c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
       0.,0.,0., 1./105 /), (/4,4/))
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

        H = H + sfactor*Kv*alpha(l)*dot_product(ex(k, :), z(l,:))
        zdz = dot_product(z(k,:),z(l,:))
        x_diff = x(k,:)-x(l,:)
        H = H - sfactor*Kv_diff*alpha(l)*zdz*2* &
             dot_product(ez(k, :), x_diff) - Kh_diff*alpha(l)*2*dot_product(ez(k, :), x_diff)
        H = H + Kh*alpha(l)*em(k)

     end do !l
  end do !k

end subroutine computeHamiltonian

subroutine testHamiltonian(sfactor,kvs,kvo,khs,kho, alpha, x, z, ex,&
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
  call computeHamiltonian(sfactor,kvs,kvo,khs,kho, alpha, &
       x, z, ex, ez, em, num_nodes, H0)
  call random_number(vx)
  vx = vx-0.5
  call computeHamiltonian(sfactor,kvs,kvo,khs,kho, &
       alpha, x+eps*vx, z, ex, ez, em, num_nodes, H1)
  test = dot_product(dex(:,1), vx(:,1)) + &
       dot_product(dex(:,2), vx(:,2)) + dot_product(dex(:,3), vx(:,3))
  print *, 'Test x', (H1-H0)/eps, -test

  call random_number(vz)
  vz = vz-0.5
  call computeHamiltonian(sfactor,kvs,kvo,khs,kho, &
       alpha, x, z+eps*vz, ex, ez, em, num_nodes, H1)
  test = dot_product(dez(:,1), vz(:,1)) + &
       dot_product(dez(:,2), vz(:,2)) + dot_product(dez(:,3), vz(:,3))
  print *, 'Test z', (H1-H0)/eps, -test
  call random_number(va)
  va = va-0.5
  call computeHamiltonian(sfactor,kvs,kvo,khs,kho, &
       alpha+eps*va, x, z, ex, ez, em, num_nodes, H1)
  test = dot_product(dea, va) 
  print *, 'Test a', (H1-H0)/eps, -test

end subroutine testHamiltonian

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

subroutine adjointSystem_NoJ(dt,sfactor,kvs,kvo,khs,kho,alpha,x,m,z,&
     dx,dm,num_times,num_nodes,ealpha,ex)
  implicit none
  integer :: num_nodes, num_times, kvo, kho
  real(8) :: sfactor, dt
  real(8) :: kvs, khs
  real(8) :: x(num_nodes, 3, num_times)
  real(8) :: m(num_nodes, num_times)
  real(8) :: z(num_nodes, 3, num_times)
  real(8) :: alpha(num_nodes)
  real(8) :: dx(num_nodes,3)
  real(8) :: dm(num_nodes)
  real(8) :: ex(num_nodes,3,num_times)
  real(8) :: ez(num_nodes,3,num_times)
  real(8) :: em(num_nodes,num_times)
  real(8) :: ealpha(num_nodes)

  real(8) :: kv_ut, kh_ut, Kv, Kv_diff, Kh, Kh_diff, lpt
  real(8) :: Kv_diff2, Kh_diff2
  real(8) :: x_diff(3), zdz
  integer :: t, k, l
  real(8) :: dex(num_nodes, 3), dez(num_nodes, 3), dea(num_nodes)
  real(8) :: c_(5, 5), c1_(4, 4), c2_(3,3)

  !f2py integer, intent(in) :: num_nodes, num_times, kvo, kho
  !f2py real(8), intent(in) :: sfactor, dt, kvs, khs
  !f2py real(8), intent(in), dimension(num_nodes) :: alpha 
  !f2py real(8), intent(in), dimension(num_nodes, 3, num_times) :: x
  !f2py real(8), intent(in), dimension(num_nodes, num_times) :: m
  !f2py real(8), intent(in), dimension(num_nodes, 3, num_times) :: z
  !f2py real(8), intent(out), dimension(num_nodes) :: ealpha 
  !f2py real(8), intent(out), dimension(num_nodes, 3, num_times) :: ex 

  ex(:,:,num_times) = dx
  em(:,num_times) = dm
  ealpha = 0
  ez(:,:,num_times) = 0
  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
  c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
       0.,0.,0., 1./105 /), (/4,4/))
  c2_ = reshape((/ 1.0/3, 1./15, 1./35,    0., 1./15, 1./35,   0.,0., 1./105 /), (/3,3/))
  

  do t=num_times,2,-1
     !$omp parallel do private(k,kv_ut,kh_ut,lpt,Kv,Kv_diff,Kh,Kh_diff, &
     !$omp& Kv_diff2,Kh_diff2, & 
     !$omp& zdz,x_diff) shared (dt,alpha,x,m,z,ex,ez,em,ealpha, &
     !$omp& dex,dez,dea,kvs,khs,kvo,kho,c_,c1_,c2_)
     do k=1,num_nodes,1
        dex(k,:) = 0
        dez(k,:) = 0
        dea(k) = 0
        do l=1,num_nodes,1

           kv_ut = norm2(x(k,:,t-1)-x(l,:,t-1)) / kvs 
           kh_ut = norm2(x(k,:,t-1)-x(l,:,t-1)) / khs 

           if (k==l) then
              Kv = 1.0
              Kv_diff = - c1_(kvo,1)/(2*kvs*kvs)
              Kh = 1.0
              Kh_diff = - c1_(kho,1)/(2*khs*khs)
              Kv_diff2 = c2_(kvo-1,1)/(4*kvs**4)
              Kh_diff2 = c2_(kho-1,1)/(4*khs**4)
           else
              lpt = c_(kvo+1, 1) + c_(kvo+1,2)*kv_ut + c_(kvo+1,3)*kv_ut**2 + c_(kvo+1,4)*kv_ut**3 + c_(kvo+1,5)*kv_ut**4
              Kv = lpt * exp(-1.0*kv_ut)
              lpt = c_(kho+1, 1) + c_(kho+1,2)*kh_ut + c_(kho+1,3)*kh_ut**2 + c_(kho+1,4)*kh_ut**3 + c_(kho+1,5)*kh_ut**4
              Kh = lpt * exp(-1.0*kh_ut)
              lpt = c1_(kvo,1) + c1_(kvo,2)*kv_ut + c1_(kvo,3)*kv_ut**2 + c1_(kvo,4)*kv_ut**3 
              Kv_diff = -lpt * exp(-1.0*kv_ut)/(2*kvs**2)
              lpt = c1_(kho,1) + c1_(kho,2)*kh_ut + c1_(kho,3)*kh_ut**2 + c1_(kho,4)*kh_ut**3 
              Kh_diff = -lpt * exp(-1.0*kh_ut)/(2*khs**2)
              lpt = c2_(kvo-1,1) + c2_(kvo-1,2)*kv_ut + c2_(kvo-1,3)*kv_ut**2 
              Kv_diff2 = lpt * exp(-1.0*kv_ut)/(4*kvs**4)
              lpt = c2_(kho-1,1) + c2_(kho-1,2)*kh_ut + c2_(kho-1,3)*kh_ut**2 
              Kh_diff2 = lpt * exp(-1.0*kh_ut)/(4*khs**4)
           end if


           x_diff = x(k,:,t-1)-x(l,:,t-1)
           zdz = dot_product(z(k,:,t-1),z(l,:,t-1))
           dex(k,:) = dex(k,:) + (-sfactor)*( &
                Kv_diff*2.0*x_diff*alpha(l)*dot_product(z(l,:,t-1),ex(k,:,t)) &
                + Kv_diff*2.0*x_diff*alpha(k)*dot_product(z(k,:,t-1),ex(l,:,t)) ) &
                + sfactor*( &
                alpha(l)*zdz*(x_diff*Kv_diff2*4*dot_product(x_diff,ez(k,:,t)) &
                +2*Kv_diff*ez(k,:,t)) &
                + alpha(k)*zdz*(-1*x_diff*Kv_diff2*4*dot_product(x_diff,ez(l,:,t)) &
                -2*Kv_diff*ez(l,:,t)) ) &
                + alpha(l)*(x_diff*Kh_diff2*4*dot_product(x_diff,ez(k,:,t)) &
                +2*Kh_diff*ez(k,:,t)) &
                + alpha(k)*(-1*x_diff*Kh_diff2*4*dot_product(x_diff,ez(l,:,t)) &
                -2*Kh_diff*ez(l,:,t)) & 
                - alpha(l)*Kh_diff*2*x_diff*em(k,t) &
                - alpha(k)*Kh_diff*2*x_diff*em(l,t) 

           dez(k,:) = dez(k,:) + (-sfactor)*Kv*alpha(k)*ex(l,:,t) & 
                + sfactor*alpha(l)*z(l,:,t-1)*Kv_diff*2*dot_product(x_diff,ez(k,:,t)) &
                + sfactor*alpha(k)*z(l,:,t-1)*Kv_diff*2*dot_product(-1*x_diff,ez(l,:,t)) 

           dea(k) = dea(k) + (-sfactor)*dot_product(z(k,:,t-1),Kv*ex(l,:,t)) & 
                + sfactor*dot_product(z(k,:,t-1),z(l,:,t-1))*Kv_diff*2* &
                dot_product(-1*x_diff,ez(l,:,t)) &
                + Kh_diff*2*dot_product(-1*x_diff,ez(l,:,t)) &
                - Kh*em(l,t) 
        end do ! l
	ex(k,:,t-1) = ex(k,:,t) - dt*dex(k,:)
	ez(k,:,t-1) = ez(k,:,t) - dt*dez(k,:)
	ealpha(k) = ealpha(k) - dt*dea(k)
	em(k,t-1) = em(k,t)
     end do ! k
     !call testHamiltonian(sfactor,kvs,kvo,khs,kho, alpha, x(:,:,t), z(:,:,t), &
      !    ex(:,:,t), ez(:,:,t), em(:,t), dex, &
       !   dez, dea, num_nodes)
     !$omp end parallel do
  end do ! t
  ! t = 1 
  ! do k=1,num_nodes,1
  !    dea(k) = 0
  !    do l=1,num_nodes,1

  !       kv_ut = norm2(x(k,:,t)-x(l,:,t)) / kvs 
  !       kh_ut = norm2(x(k,:,t)-x(l,:,t)) / khs 

  !       if (k==l) then
  !          Kv = 1.0
  !          Kv_diff = -1.0/((2*kvo-1)*2*kvs*kvs)
  !          Kh = 1.0
  !          Kh_diff = -1.0/((2*kho-1)*2*khs*khs)
  !       else
  !          lpt = (105 + 105*kv_ut + 45*kv_ut**2 + 10*kv_ut**3 + kv_ut**4)/105
  !          Kv = lpt * exp(-1.0*kv_ut)
  !          lpt = (105 + 105*kh_ut + 45*kh_ut**2 + 10*kh_ut**3 + kh_ut**4)/105
  !          Kh = lpt * exp(-1.0*kh_ut)
  !          lpt = (15 + 15*kv_ut + 6*kv_ut**2 + kv_ut**3)/105
  !          Kv_diff = -lpt * exp(-1.0*kv_ut)/(2*kvs**2)
  !          lpt = (15 + 15*kh_ut + 6*kh_ut**2 + kh_ut**3)/105
  !          Kh_diff = -lpt * exp(-1.0*kh_ut)/(2*khs**2)
  !       end if

  !       x_diff = x(k,:,t)-x(l,:,t)
  !       dea(k) = dea(k) + (-sfactor)*dot_product(z(k,:,t),Kv*ex(l,:,t)) & 
  !            + sfactor*dot_product(z(k,:,t),z(l,:,t))*Kv_diff*2* &
  !            dot_product(-1*x_diff,ez(l,:,t)) &
  !            + Kh_diff*2*dot_product(-1*x_diff,ez(l,:,t)) &
  !            - Kh*em(l,t) 
  !    end do ! l
  !    ealpha(k) = ealpha(k) - dt*dea(k)
  ! end do ! k
end subroutine adjointSystem_NoJ



subroutine adjointSystem_unconstrained(dt,sfactor,kvs,kvo,khs,kho,alpha,x,m,z,&
     dx,dm,num_times,num_nodes,ealpha,ex, ez0)
  implicit none
  integer :: num_nodes, num_times, kvo, kho
  real(8) :: sfactor, dt
  real(8) :: kvs, khs
  real(8) :: x(num_nodes, 3, num_times)
  real(8) :: m(num_nodes, num_times)
  real(8) :: z(num_nodes, 3, num_times)
  real(8) :: alpha(num_nodes)
  real(8) :: dx(num_nodes,3)
  real(8) :: dm(num_nodes)
  real(8) :: ex(num_nodes,3,num_times)
  real(8) :: ez0(num_nodes,3)
  real(8) :: em(num_nodes,num_times)
  real(8) :: ealpha(num_nodes)

  real(8) :: ez(num_nodes,3,num_times)
  real(8) :: kv_ut, kh_ut, Kv, Kv_diff, Kh, Kh_diff, lpt
  real(8) :: Kv_diff2, Kh_diff2
  real(8) :: x_diff(3), zdz
  integer :: t, k, l
  real(8) :: dex(num_nodes, 3), dez(num_nodes, 3), dea(num_nodes)
  real(8) :: c_(5, 5), c1_(4, 4), c2_(3,3)

  !f2py integer, intent(in) :: num_nodes, num_times, kvo, kho
  !f2py real(8), intent(in) :: sfactor, dt, kvs, khs
  !f2py real(8), intent(in), dimension(num_nodes) :: alpha 
  !f2py real(8), intent(in), dimension(num_nodes, 3, num_times) :: x
  !f2py real(8), intent(in), dimension(num_nodes, num_times) :: m
  !f2py real(8), intent(in), dimension(num_nodes, 3, num_times) :: z
  !f2py real(8), intent(out), dimension(num_nodes) :: ealpha 
  !f2py real(8), intent(out), dimension(num_nodes, 3, num_times) :: ex 
  !f2py real(8), intent(out), dimension(num_nodes, 3) :: ez0 

  ex(:,:,num_times) = dx
  em(:,num_times) = dm
  ealpha = 0
  ez(:,:,num_times) = 0
  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
  c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
       0.,0.,0., 1./105 /), (/4,4/))
  c2_ = reshape((/ 1.0/3, 1./15, 1./35,    0., 1./15, 1./35,   0.,0., 1./105 /), (/3,3/))
  !print *, c_(kho+1,:)

  do t=num_times,2,-1
     !$omp parallel do private(k,kv_ut,kh_ut,lpt,Kv,Kv_diff,Kh,Kh_diff, &
     !$omp& Kv_diff2,Kh_diff2, & 
     !$omp& zdz,x_diff) shared (dt,alpha,x,m,z,ex,ez,em,ealpha, &
     !$omp& kvs,khs,kvo,kho,dex,dez,dea,c_,c1_,c2_)
     do k=1,num_nodes,1
        dex(k,:) = 0
        dez(k,:) = 0
        dea(k) = 0
        do l=1,num_nodes,1

           x_diff = x(k,:,t-1)-x(l,:,t-1)
           zdz = dot_product(z(k,:,t-1),z(l,:,t-1))
           kv_ut = norm2(x_diff) 
           kh_ut = kv_ut / khs 
           kv_ut = kv_ut / kvs 

           if (k==l) then
              Kv = 1.0
              Kv_diff = - c1_(kvo,1)/(2*kvs*kvs)
              Kh = 1.0
              Kh_diff = - c1_(kho,1)/(2*khs*khs)
              Kv_diff2 = c2_(kvo-1,1)/(4*kvs**4)
              Kh_diff2 = c2_(kho-1,1)/(4*khs**4)
           else
              lpt = c_(kvo+1, 1) + c_(kvo+1,2)*kv_ut + c_(kvo+1,3)*kv_ut**2 + c_(kvo+1,4)*kv_ut**3 + c_(kvo+1,5)*kv_ut**4
              Kv = lpt * exp(-1.0*kv_ut)
              lpt = c_(kho+1, 1) + c_(kho+1,2)*kh_ut + c_(kho+1,3)*kh_ut**2 + c_(kho+1,4)*kh_ut**3 + c_(kho+1,5)*kh_ut**4
              Kh = lpt * exp(-1.0*kh_ut)
              lpt = c1_(kvo,1) + c1_(kvo,2)*kv_ut + c1_(kvo,3)*kv_ut**2 + c1_(kvo,4)*kv_ut**3 
              Kv_diff = -lpt * exp(-1.0*kv_ut)/(2*kvs**2)
              lpt = c1_(kho,1) + c1_(kho,2)*kh_ut + c1_(kho,3)*kh_ut**2 + c1_(kho,4)*kh_ut**3 
              Kh_diff = -lpt * exp(-1.0*kh_ut)/(2*khs**2)
              lpt = c2_(kvo-1,1) + c2_(kvo-1,2)*kv_ut + c2_(kvo-1,3)*kv_ut**2 
              Kv_diff2 = lpt * exp(-1.0*kv_ut)/(4*kvs**4)
              lpt = c2_(kho-1,1) + c2_(kho-1,2)*kh_ut + c2_(kho-1,3)*kh_ut**2 
              Kh_diff2 = lpt * exp(-1.0*kh_ut)/(4*khs**4)
           end if

           dex(k,:) = dex(k,:) + (-sfactor)*( &
                Kv_diff*2.0*x_diff*dot_product(z(l,:,t-1),ex(k,:,t)) &
                + Kv_diff*2.0*x_diff*dot_product(z(k,:,t-1),ex(l,:,t)) ) &
                + sfactor*( &
                zdz*(x_diff*Kv_diff2*4*dot_product(x_diff,ez(k,:,t)) &
                +2*Kv_diff*ez(k,:,t)) &
                + zdz*(-1*x_diff*Kv_diff2*4*dot_product(x_diff,ez(l,:,t)) &
                -2*Kv_diff*ez(l,:,t)) ) &
                + alpha(k)*alpha(l)*(x_diff*Kh_diff2*4*dot_product(x_diff,ez(k,:,t)) &
                +2*Kh_diff*ez(k,:,t)) &
                + alpha(k)*alpha(l)*(-1*x_diff*Kh_diff2*4*dot_product(x_diff,ez(l,:,t)) &
                -2*Kh_diff*ez(l,:,t)) & 
                - alpha(l)*Kh_diff*2*x_diff*em(k,t) &
                - alpha(k)*Kh_diff*2*x_diff*em(l,t) 

           dez(k,:) = dez(k,:) + (-sfactor)*Kv*ex(l,:,t) & 
                + sfactor*z(l,:,t-1)*Kv_diff*2*dot_product(x_diff,ez(k,:,t)) &
                + sfactor*z(l,:,t-1)*Kv_diff*2*dot_product(-1*x_diff,ez(l,:,t)) 

           dea(k) = dea(k) + Kh_diff*alpha(l)*2*dot_product(-1*x_diff,ez(l,:,t)) &
                + Kh_diff*alpha(l)*2*dot_product(x_diff,ez(k,:,t)) - Kh*em(l,t) 
        end do ! l
	ex(k,:,t-1) = ex(k,:,t) - dt*dex(k,:)
	ez(k,:,t-1) = ez(k,:,t) - dt*dez(k,:)
	ealpha(k) = ealpha(k) - dt*dea(k)
	em(k,t-1) = em(k,t)
     end do ! k
     ! call testHamiltonian_u(sfactor,kvs,kvo,khs,kho, alpha, x(:,:,t-1), z(:,:,t-1), &
     !      ex(:,:,t), ez(:,:,t), em(:,t), dex, &
     !      dez, dea, num_nodes)
     !$omp end parallel do
  end do ! t
  ez0 = ez(:,:,1)
  ! t = 1 
  ! do k=1,num_nodes,1
  !    dea(k) = 0
  !    dez(k,:) = 0
  !    do l=1,num_nodes,1

  !       kv_ut = norm2(x(k,:,t)-x(l,:,t)) / kvs 
  !       kh_ut = norm2(x(k,:,t)-x(l,:,t)) / khs 

  !          if (k==l) then
  !             Kv = 1.0
  !             Kv_diff = -1.0/((2*kvo-1)*2*kvs*kvs)
  !             Kh = 1.0
  !             Kh_diff = -1.0/((2*kho-1)*2*khs*khs)
  !          else
  !             lpt = (105 + 105*kv_ut + 45*kv_ut**2 + 10*kv_ut**3 + kv_ut**4)/105
  !             Kv = lpt * exp(-1.0*kv_ut)
  !             lpt = (105 + 105*kh_ut + 45*kh_ut**2 + 10*kh_ut**3 + kh_ut**4)/105
  !             Kh = lpt * exp(-1.0*kh_ut)
  !             lpt = (15 + 15*kv_ut + 6*kv_ut**2 + kv_ut**3)/105
  !             Kv_diff = -lpt * exp(-1.0*kv_ut)/(2*kvs**2)
  !             lpt = (15 + 15*kh_ut + 6*kh_ut**2 + kh_ut**3)/105
  !             Kh_diff = -lpt * exp(-1.0*kh_ut)/(2*khs**2)
  !          end if

  !       x_diff = x(k,:,t)-x(l,:,t)
  !       dez(k,:) = dez(k,:) + (-sfactor)*Kv*ex(l,:,t) & 
  !            + sfactor*z(l,:,t)*Kv_diff*2*dot_product(x_diff,ez(k,:,t)) &
  !            + sfactor*z(l,:,t)*Kv_diff*2*dot_product(-1*x_diff,ez(l,:,t)) 
  !       dea(k) = dea(k) + Kh_diff*alpha(l)*2*dot_product(-1*x_diff,ez(l,:,t)) &
  !            - Kh*em(l,t) 
  !    end do ! l
  !    ealpha(k) = ealpha(k) - dt*dea(k)
  !    ez0(k, :) = ez(k, :,t) - dt * dez(k,:)
  ! end do ! k
end subroutine adjointSystem_unconstrained

subroutine applyK(x, y, beta, sig, ord, num_nodes, f)
  implicit none
  integer :: num_nodes
  real(8) :: x(num_nodes, 3)
  real(8) :: y(num_nodes, 3)
  real(8) :: sig
  integer :: ord
  real(8) :: f(num_nodes)
  real(8) :: beta(num_nodes)
  real(8) :: Kh

  !f2py integer, intent(in) :: num_nodes
  !f2py real(8), intent(in), dimension(num_nodes, 3) :: x 
  !f2py real(8), intent(in), dimension(num_nodes, 3) :: y 
  !f2py real(8), intent(in), dimension(num_nodes) :: beta 
  !f2py real(8), intent(in) :: sig
  !f2py integer, intent(in) :: ord
  !f2py real(8), intent(out), dimension(num_nodes) :: f

  real(8) :: ut
  real(8) :: lpt
  integer :: k,l 
  real(8) :: df
  real(8) :: c_(5, 5)
  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))

  !$omp parallel do private(k,l,ut,lpt,Kh,df) shared &
  !$omp& (num_nodes, f, sig, ord, beta, c_)
  do k = 1, num_nodes, 1
     df = 0
     do l = 1, num_nodes, 1
        ut = sqrt((x(k,1)-y(l,1))**2 + (x(k,2)-y(l,2))**2) / sig
        if (ut < 1e-8) then
           Kh = 1.0
        else
           lpt = c_(ord+1, 1) + c_(ord+1,2)*ut + c_(ord+1,3)*ut**2 + c_(ord+1,4)*ut**3 + c_(ord+1,5)*ut**4
           Kh = lpt * exp(-1.0*ut)
        end if
        df = df + Kh * beta(l)
     end do
     f(k) = df
  end do
  !$omp end parallel do
end subroutine applyK

subroutine applyK_and_Diff(x, y, beta, sig, ord, num_nodes, f, f2)
  implicit none
  integer :: num_nodes
  real(8) :: x(num_nodes, 3)
  real(8) :: y(num_nodes, 3)
  real(8) :: sig
  integer :: ord
  real(8) :: f(num_nodes)
  real(8) :: f2(num_nodes,3)
  real(8) :: beta(num_nodes)
  real(8) :: Kh, Kh_diff

  !f2py integer, intent(in) :: num_nodes
  !f2py real(8), intent(in), dimension(num_nodes, 3) :: x 
  !f2py real(8), intent(in), dimension(num_nodes, 3) :: y 
  !f2py real(8), intent(in), dimension(num_nodes) :: beta 
  !f2py real(8), intent(in) :: sig
  !f2py integer, intent(in) :: ord
  !f2py real(8), intent(out), dimension(num_nodes) :: f
  !f2py real(8), intent(out), dimension(num_nodes,3) :: f2

  real(8) :: ut
  real(8) :: lpt
  integer :: k,l 
  real(8) :: df, df2(3)
  real(8) :: c_(5, 5), c1_(4, 4)
  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
  c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
       0.,0.,0., 1./105 /), (/4,4/))

  !$omp parallel do private(k,l,ut,lpt,Kh,Kh_diff,df,df2) shared &
  !$omp& (num_nodes, f, f2, sig, ord, beta, c_, c1_)
  do k = 1, num_nodes, 1
     df = 0
     df2 = 0
     do l = 1, num_nodes, 1
        ut = sqrt((x(k,1)-y(l,1))**2 + (x(k,2)-y(l,2))**2) / sig
        if (ut < 1e-8) then
           Kh = 1.0
           Kh_diff = - c1_(ord,1)/(2*sig*sig)
        else
           lpt = c_(ord+1, 1) + c_(ord+1,2)*ut + c_(ord+1,3)*ut**2 + c_(ord+1,4)*ut**3 + c_(ord+1,5)*ut**4
           Kh = lpt * exp(-1.0*ut)
           lpt = c1_(ord,1) + c1_(ord,2)*ut + c1_(ord,3)*ut**2 + c1_(ord,4)*ut**3 
           Kh_diff = -lpt * exp(-1.0*ut)/(2*sig**2)
        end if
        df = df + Kh * beta(l)
        df2 = df2 + Kh_diff * 2*(x(k,:)-y(l,:))* beta(l)
     end do
     f(k) = f(k) + df
     f2(k,:) = f2(k,:) + df2
  end do
  !$omp end parallel do
end subroutine applyK_and_Diff
