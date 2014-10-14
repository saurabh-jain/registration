subroutine applyk(x, y, beta, sig, ord, num_nodes, num_nodes_y, dim, dimb, f)
  implicit none
  integer :: num_nodes, num_nodes_y, dim, dimb
  real(8) :: x(num_nodes, dim)
  real(8) :: y(num_nodes_y, dim)
  real(8) :: sig
  integer :: ord
  real(8) :: f(num_nodes, dimb)
  real(8) :: beta(num_nodes_y, dimb)
  real(8) :: Kh

  !f2py integer, intent(in) :: num_nodes, num_nodes_y, dim, dimb
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: x 
  !f2py real(8), intent(in), dimension(num_nodes_y, dim) :: y 
  !f2py real(8), intent(in), dimension(num_nodes_y, dimb) :: beta 
  !f2py real(8), intent(in) :: sig
  !f2py integer, intent(in) :: ord
  !f2py real(8), intent(out), dimension(num_nodes, dimb) :: f

  real(8) :: ut
  real(8) :: lpt
  integer :: k,l 
  real(8) :: df(dimb)
  real(8) :: c_(5, 5)
  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))

  !$omp parallel do private(k,l,ut,lpt,Kh,df) shared &
  !$omp& (num_nodes, num_nodes_y, f, sig, ord, beta, c_)
  do k = 1, num_nodes, 1
     df = 0
     do l = 1, num_nodes_y, 1
        ut = norm2(x(k,:) - y(l,:)) / sig
        if (ord > 4) then
           ut = ut * ut
           if (ut < 1e-8) then
              Kh = 1.0
           else
              Kh = exp(-0.5*ut)
           end if
        else           
           if (ut < 1e-8) then
              Kh = 1.0
           else
              lpt = c_(ord+1, 1) + c_(ord+1,2)*ut + c_(ord+1,3)*ut**2 + c_(ord+1,4)*ut**3 + c_(ord+1,5)*ut**4
              Kh = lpt * exp(-1.0*ut)
           end if
        end if
        df = df + Kh * beta(l,:)
     end do
     f(k,:) = df
  end do
  !$omp end parallel do
end subroutine applyK

subroutine applykdifft(x, y, a1, a2, sig, ord, num_nodes, num_nodes_y, dim, dimb, na, f)
  implicit none
  integer :: num_nodes, num_nodes_y, dim, dimb, na
  real(8) :: x(num_nodes, dim)
  real(8) :: y(num_nodes_y, dim)
  real(8) :: sig
  integer :: ord
  real(8) :: f(num_nodes, dim)
  real(8) :: a1(na, num_nodes, dimb)
  real(8) :: a2(na, num_nodes_y, dimb)

  real(8) :: Kh_diff

  !f2py integer, intent(in) :: num_nodes, num_nodes_y, dim, dimb, na
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: x 
  !f2py real(8), intent(in), dimension(num_nodes_y, dim) :: y 
  !f2py real(8), intent(in), dimension(na,num_nodes, dimb) :: a1
  !f2py real(8), intent(in), dimension(na,num_nodes_y, dimb) :: a2 
  !f2py real(8), intent(in) :: sig
  !f2py integer, intent(in) :: ord
  !f2py real(8), intent(out), dimension(num_nodes, dim) :: f

  real(8) :: ut
  real(8) :: lpt
  integer :: k,l 
  real(8) :: df(dim)
  real(8) :: c_(5, 5), c1_(4, 4)
  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
  c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
       0.,0.,0., 1./105 /), (/4,4/))

  !$omp parallel do private(k,l,ut,lpt,Kh_diff,df) shared &
  !$omp& (num_nodes, num_nodes_y, f, sig, ord, a1, a2, c_, c1_)
  do k = 1, num_nodes, 1
     df = 0
     do l = 1, num_nodes_y, 1
        ut = norm2(x(k,:) - y(l,:)) / sig
        if (ord > 4) then
           ut = ut * ut
           if (ut < 1e-8) then
              Kh_diff = - 1.0/(2*sig**2)
           else
              Kh_diff = (-1.0)* exp(-0.5*ut)/(2*sig**2)
           end if
        else           
           if (ut < 1e-8) then
              Kh_diff = - c1_(ord,1)/(2*sig*sig)
           else
              lpt = c1_(ord,1) + c1_(ord,2)*ut + c1_(ord,3)*ut**2 + c1_(ord,4)*ut**3 
              Kh_diff = -lpt * exp(-1.0*ut)/(2*sig**2)
           end if
        end if
        df = df + Kh_diff * 2*(x(k,:)-y(l,:))* sum(a1(:,k,:)*a2(:,l,:)) ;
     end do
     f(k, :) = df
  end do
  !$omp end parallel do
end subroutine applykdifft

subroutine applykdiff1(x, a1, a2, sig, ord, num_nodes,dim, f)
  implicit none
  integer :: num_nodes, dim
  real(8) :: x(num_nodes, dim)
  real(8) :: sig
  integer :: ord
  real(8) :: f(num_nodes, dim)
  real(8) :: a1(num_nodes, dim)
  real(8) :: a2(num_nodes, dim)

  real(8) :: Kh_diff

  !f2py integer, intent(in) :: num_nodes, num_nodes, dim
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: x 
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: a1
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: a2 
  !f2py real(8), intent(in) :: sig
  !f2py integer, intent(in) :: ord
  !f2py real(8), intent(out), dimension(num_nodes, dim) :: f

  real(8) :: ut
  real(8) :: lpt
  integer :: k,l 
  real(8) :: df(dim)
  real(8) :: c_(5, 5), c1_(4, 4)
  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
  c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
       0.,0.,0., 1./105 /), (/4,4/))

  !$omp parallel do private(k,l,ut,lpt,Kh_diff,df) shared &
  !$omp& (num_nodes, f, sig, ord, a1, a2, c_, c1_)
  do k = 1, num_nodes, 1
     df = 0
     do l = 1, num_nodes, 1
        ut = norm2(x(k,:) - x(l,:)) / sig
        if (ord > 4) then
           ut = ut * ut
           if (ut < 1e-8) then
              Kh_diff = - 1.0/(2*sig**2)
           else
              Kh_diff = (-1.0)* exp(-0.5*ut)/(2*sig**2)
           end if
        else           
           if (ut < 1e-8) then
              Kh_diff = - c1_(ord,1)/(2*sig*sig)
           else
              lpt = c1_(ord,1) + c1_(ord,2)*ut + c1_(ord,3)*ut**2 + c1_(ord,4)*ut**3 
              Kh_diff = -lpt * exp(-1.0*ut)/(2*sig**2)
           end if
        end if
        df = df + Kh_diff * 2*sum((x(k,:)-x(l,:))*a1(k,:)) * a2(l,:) ;
     end do
     f(k, :) = f(k, :) + df
  end do
  !$omp end parallel do
end subroutine applykdiff1

subroutine applykdiff2(x, a1, a2, sig, ord, num_nodes, dim, f)
  implicit none
  integer :: num_nodes, dim
  real(8) :: x(num_nodes, dim)
  real(8) :: sig
  integer :: ord
  real(8) :: f(num_nodes, dim)
  real(8) :: a1(num_nodes, dim)
  real(8) :: a2(num_nodes, dim)

  real(8) :: Kh_diff

  !f2py integer, intent(in) :: num_nodes, dim
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: x 
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: a1
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: a2 
  !f2py real(8), intent(in) :: sig
  !f2py integer, intent(in) :: ord
  !f2py real(8), intent(out), dimension(num_nodes, dim) :: f

  real(8) :: ut
  real(8) :: lpt
  integer :: k,l 
  real(8) :: df(dim)
  real(8) :: c_(5, 5), c1_(4, 4)
  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
  c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
       0.,0.,0., 1./105 /), (/4,4/))

  !$omp parallel do private(k,l,ut,lpt,Kh_diff,df) shared &
  !$omp& (num_nodes, f, sig, ord, a1, a2, c_, c1_)
  do k = 1, num_nodes, 1
     df = 0
     do l = 1, num_nodes, 1
        ut = norm2(x(k,:) - x(l,:)) / sig
        if (ord > 4) then
           ut = ut * ut
           if (ut < 1e-8) then
              Kh_diff = - 1.0/(2*sig**2)
           else
              Kh_diff = (-1.0)* exp(-0.5*ut)/(2*sig**2)
           end if
        else           
           if (ut < 1e-8) then
              Kh_diff = - c1_(ord,1)/(2*sig*sig)
           else
              lpt = c1_(ord,1) + c1_(ord,2)*ut + c1_(ord,3)*ut**2 + c1_(ord,4)*ut**3 
              Kh_diff = -lpt * exp(-1.0*ut)/(2*sig**2)
           end if
        end if
        df = df - Kh_diff * 2*sum((x(k,:)-x(l,:))*a1(l,:)) * a2(l,:) ;
     end do
     f(k, :) = f(k, :) + df
  end do
  !$omp end parallel do
end subroutine applykdiff2


subroutine applykdiff1and2(x, a1, a2, sig, ord, num_nodes,dim, f)
  implicit none
  integer :: num_nodes, dim
  real(8) :: x(num_nodes, dim)
  real(8) :: sig
  integer :: ord
  real(8) :: f(num_nodes, dim)
  real(8) :: a1(num_nodes, dim)
  real(8) :: a2(num_nodes, dim)

  real(8) :: Kh_diff

  !f2py integer, intent(in) :: num_nodes, dim
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: x 
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: a1
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: a2 
  !f2py real(8), intent(in) :: sig
  !f2py integer, intent(in) :: ord
  !f2py real(8), intent(out), dimension(num_nodes, dim) :: f

  real(8) :: ut
  real(8) :: lpt
  integer :: k,l 
  real(8) :: df(dim), dx(dim)
  real(8) :: c_(5, 5), c1_(4, 4)
  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
  c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
       0.,0.,0., 1./105 /), (/4,4/))

  !$omp parallel do private(k,l,ut,lpt,Kh_diff,dx,df) shared &
  !$omp& (num_nodes, x, f, sig, ord, a1, a2, c_, c1_)
  do k = 1, num_nodes, 1
     df = 0
     do l = 1, num_nodes, 1
        dx=x(k,:) - x(l,:)
        ut = norm2(dx) / sig
        if (ord > 4) then
           ut = ut * ut
           if (ut < 1e-8) then
              Kh_diff = - 1.0/(2*sig**2)
           else
              Kh_diff = (-1.0)* exp(-0.5*ut)/(2*sig**2)
           end if
        else           
           if (ut < 1e-8) then
              Kh_diff = - c1_(ord,1)/(2*sig*sig)
           else
              lpt = c1_(ord,1) + c1_(ord,2)*ut + c1_(ord,3)*ut**2 + c1_(ord,4)*ut**3 
              Kh_diff = -lpt * exp(-1.0*ut)/(2*sig**2)
           end if
        end if
        df = df + Kh_diff * 2*sum(dx*(a1(k,:)-a1(l,:))) * a2(l,:) ;
     end do
     f(k, :) = df
  end do
  !$omp end parallel do
end subroutine applykdiff1and2


subroutine applykdiff11(x, a1, a2, p, sig, ord, num_nodes, dim, f)
  implicit none
  integer :: num_nodes, num_nodes_y, dim
  real(8) :: x(num_nodes, dim)
  real(8) :: y(num_nodes, dim)
  real(8) :: sig
  integer :: ord
  real(8) :: f(num_nodes, dim)
  real(8) :: a1(num_nodes, dim)
  real(8) :: a2(num_nodes, dim)
  real(8) :: p(num_nodes, dim)

  real(8) :: Kh_diff, Kh_diff2

  !f2py integer, intent(in) :: num_nodes, num_nodes, dim
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: x 
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: a1
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: a2 
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: p 
  !f2py real(8), intent(in) :: sig
  !f2py integer, intent(in) :: ord
  !f2py real(8), intent(out), dimension(num_nodes, dim) :: f

  real(8) :: ut
  real(8) :: lpt
  integer :: k,l 
  real(8) :: df(dim)
  real(8) :: c_(5, 5), c1_(4, 4), c2_(3,3)
  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
  c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
       0.,0.,0., 1./105 /), (/4,4/))
  c2_ = reshape((/ 1.0/3, 1./15, 1./35,    0., 1./15, 1./35,   0.,0., 1./105 /), (/3,3/))

  !$omp parallel do private(k,l,ut,lpt,Kh_diff,df) shared &
  !$omp& (num_nodes, num_nodes_y, f, sig, ord, a1, a2, c_, c1_)
  do k = 1, num_nodes, 1
     df = 0
     do l = 1, num_nodes_y, 1
        ut = norm2(x(k,:) - y(l,:)) / sig
        if (ord > 4) then
           ut = ut * ut
           if (ut < 1e-8) then
              Kh_diff = - 1.0/(2*sig**2)
              Kh_diff2 = 1.0/(4*sig**4)
           else
              Kh_diff = (-1.0)* exp(-0.5*ut)/(2*sig**2)
              Kh_diff2 = exp(-0.5*ut)/(4*sig**4)
           end if
        else           
           if (ut < 1e-8) then
              Kh_diff = - c1_(ord,1)/(2*sig*sig)
              Kh_diff2 = c2_(ord-1,1)/(4*sig**4)
           else
              lpt = c1_(ord,1) + c1_(ord,2)*ut + c1_(ord,3)*ut**2 + c1_(ord,4)*ut**3 
              Kh_diff = -lpt * exp(-1.0*ut)/(2*sig**2)
              lpt = c2_(ord-1,1) + c2_(ord-1,2)*ut + c2_(ord-1,3)*ut**2 
              Kh_diff2 = lpt * exp(-1.0*ut)/(4*sig**4)
           end if
        end if
        df = df + Kh_diff2 * 4*sum(a1(k,:) * a2(l,:)) * sum((x(k,:)-y(l,:))*p(k,:)) *(x(k,:)-y(l,:)) &
             & + Kh_diff * 2 * sum(a1(k,:) * a2(l,:)) * p(k,:)
     end do
     f(k, :) = f(k, :) + df
  end do
  !$omp end parallel do
end subroutine applykdiff11

subroutine applykdiff12(x, a1, a2, p, sig, ord, num_nodes, dim, f)
  implicit none
  integer :: num_nodes, num_nodes_y, dim
  real(8) :: x(num_nodes, dim)
  real(8) :: y(num_nodes, dim)
  real(8) :: sig
  integer :: ord
  real(8) :: f(num_nodes, dim)
  real(8) :: a1(num_nodes, dim)
  real(8) :: a2(num_nodes, dim)
  real(8) :: p(num_nodes, dim)

  real(8) :: Kh_diff, Kh_diff2

  !f2py integer, intent(in) :: num_nodes, num_nodes, dim
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: x 
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: a1
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: a2 
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: p 
  !f2py real(8), intent(in) :: sig
  !f2py integer, intent(in) :: ord
  !f2py real(8), intent(out), dimension(num_nodes, dim) :: f

  real(8) :: ut
  real(8) :: lpt
  integer :: k,l 
  real(8) :: df(dim)
  real(8) :: c_(5, 5), c1_(4, 4), c2_(3,3)
  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
  c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
       0.,0.,0., 1./105 /), (/4,4/))
  c2_ = reshape((/ 1.0/3, 1./15, 1./35,    0., 1./15, 1./35,   0.,0., 1./105 /), (/3,3/))

  !$omp parallel do private(k,l,ut,lpt,Kh_diff,df) shared &
  !$omp& (num_nodes, num_nodes_y, f, sig, ord, a1, a2, c_, c1_)
  do k = 1, num_nodes, 1
     df = 0
     do l = 1, num_nodes_y, 1
        ut = norm2(x(k,:) - y(l,:)) / sig
        if (ord > 4) then
           ut = ut * ut
           if (ut < 1e-8) then
              Kh_diff = - 1.0/(2*sig**2)
              Kh_diff2 = 1.0/(4*sig**4)
           else
              Kh_diff = (-1.0)* exp(-0.5*ut)/(2*sig**2)
              Kh_diff2 = exp(-0.5*ut)/(4*sig**4)
           end if
        else           
           if (ut < 1e-8) then
              Kh_diff = - c1_(ord,1)/(2*sig*sig)
              Kh_diff2 = c2_(ord-1,1)/(4*sig**4)
           else
              lpt = c1_(ord,1) + c1_(ord,2)*ut + c1_(ord,3)*ut**2 + c1_(ord,4)*ut**3 
              Kh_diff = -lpt * exp(-1.0*ut)/(2*sig**2)
              lpt = c2_(ord-1,1) + c2_(ord-1,2)*ut + c2_(ord-1,3)*ut**2 
              Kh_diff2 = lpt * exp(-1.0*ut)/(4*sig**4)
           end if
        end if
        df = df - Kh_diff2 * 4*sum(a1(k,:) * a2(l,:)) &
             &* sum((x(k,:)-y(l,:))*p(l,:)) *(x(k,:)-y(l,:)) &
             & - Kh_diff * 2 * sum(a1(k,:) * a2(l,:)) * p(l,:)
     end do
     f(k, :) = f(k, :) + df
  end do
  !$omp end parallel do
end subroutine applykdiff12

subroutine applykdiff11and12(x, a1, a2, p, sig, ord, num_nodes, dim, f)
  implicit none
  integer :: num_nodes, dim
  real(8) :: x(num_nodes, dim)
  real(8) :: sig
  integer :: ord
  real(8) :: f(num_nodes, dim)
  real(8) :: a1(num_nodes, dim)
  real(8) :: a2(num_nodes, dim)
  real(8) :: p(num_nodes, dim)

  real(8) :: Kh_diff, Kh_diff2

  !f2py integer, intent(in) :: num_nodes, dim
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: x 
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: a1
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: a2 
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: p 
  !f2py real(8), intent(in) :: sig
  !f2py integer, intent(in) :: ord
  !f2py real(8), intent(out), dimension(num_nodes, dim) :: f

  real(8) :: ut
  real(8) :: lpt
  integer :: k,l 
  real(8) :: df(dim), dx(dim), dp(dim)
  real(8) :: c_(5, 5), c1_(4, 4), c2_(3,3)
  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
  c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
       0.,0.,0., 1./105 /), (/4,4/))
  c2_ = reshape((/ 1.0/3, 1./15, 1./35,    0., 1./15, 1./35,   0.,0., 1./105 /), (/3,3/))

  !$omp parallel do private(k,l,ut,lpt,Kh_diff,Kh_diff2,df,dx,dp) shared &
  !$omp& (num_nodes, x, p, f, sig, ord, a1, a2, c2_, c1_)
  do k = 1, num_nodes, 1
     df = 0
     do l = 1, num_nodes, 1
        dx = x(k,:) - x(l,:)
        dp = p(k,:) - p(l,:)
        ut = norm2(dx) / sig
        if (ord > 4) then
           ut = ut * ut
           if (ut < 1e-8) then
              Kh_diff = - 1.0/(2*sig**2)
              Kh_diff2 = 1.0/(4*sig**4)
           else
              Kh_diff = (-1.0)* exp(-0.5*ut)/(2*sig**2)
              Kh_diff2 = exp(-0.5*ut)/(4*sig**4)
           end if
        else           
           if (ut < 1e-8) then
              Kh_diff = - c1_(ord,1)/(2*sig*sig)
              Kh_diff2 = c2_(ord-1,1)/(4*sig**4)
           else
              lpt = c1_(ord,1) + c1_(ord,2)*ut + c1_(ord,3)*ut**2 + c1_(ord,4)*ut**3 
              Kh_diff = -lpt * exp(-1.0*ut)/(2*sig**2)
              lpt = c2_(ord-1,1) + c2_(ord-1,2)*ut + c2_(ord-1,3)*ut**2 
              Kh_diff2 = lpt * exp(-1.0*ut)/(4*sig**4)
           end if
        end if
        df = df + 2 * sum(a1(k,:)*a2(l,:)) *  (2 * Kh_diff2 *sum(dx*dp) *dx + Kh_diff * dp)
     end do
     f(k, :) = df
  end do
  !$omp end parallel do
end subroutine applykdiff11and12


subroutine applykmat(x, y, beta, sig, ord, num_nodes, num_nodes_y, dim, dimb, f)
  implicit none
  integer :: num_nodes, num_nodes_y, dim, dimb
  real(8) :: x(num_nodes, dim)
  real(8) :: y(num_nodes_y, dim)
  real(8) :: sig
  integer :: ord
  real(8) :: f(num_nodes,dimb)
  real(8) :: beta(num_nodes, num_nodes_y,dimb)
  real(8) :: Kh

  !f2py integer, intent(in) :: num_nodes, num_nodes_y, dim, dimb
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: x 
  !f2py real(8), intent(in), dimension(num_nodes_y, dim) :: y 
  !f2py real(8), intent(in), dimension(num_nodes, num_nodes_y,dimb) :: beta 
  !f2py real(8), intent(in) :: sig
  !f2py integer, intent(in) :: ord
  !f2py real(8), intent(out), dimension(num_nodes,dimb) :: f

  real(8) :: ut
  real(8) :: lpt
  integer :: k,l 
  real(8) :: df(dimb)
  real(8) :: c_(5, 5)
  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))

  !$omp parallel do private(k,l,ut,lpt,Kh,df) shared &
  !$omp& (num_nodes, num_nodes_y, f, sig, ord, beta, c_)
  do k = 1, num_nodes, 1
     df = 0 
     do l = 1, num_nodes_y, 1
        ut = norm2(x(k,:) - y(l,:)) / sig
        if (ord > 4) then
           ut = ut * ut
           if (ut < 1e-8) then
              Kh = 1.0
           else
              Kh = exp(-0.5*ut)
           end if
        else           
           if (ut < 1e-8) then
              Kh = 1.0
           else
              lpt = c_(ord+1, 1) + c_(ord+1,2)*ut + c_(ord+1,3)*ut**2 + c_(ord+1,4)*ut**3 + c_(ord+1,5)*ut**4
              Kh = lpt * exp(-1.0*ut)
           end if
        end if
        df = df + Kh * beta(k,l,:)
     end do
     f(k,:) = df
  end do
  !$omp end parallel do
end subroutine applyKmat

subroutine applykdiffmat(x, y, beta, sig, ord, num_nodes, num_nodes_y, dim, f)
  implicit none
  integer :: num_nodes, num_nodes_y, dim
  real(8) :: x(num_nodes, dim)
  real(8) :: y(num_nodes_y, dim)
  real(8) :: sig
  integer :: ord
  real(8) :: f(num_nodes, dim)
  real(8) :: beta(num_nodes, num_nodes_y)

  real(8) :: Kh_diff

  !f2py integer, intent(in) :: num_nodes, num_nodes_y, dim
  !f2py real(8), intent(in), dimension(num_nodes, dim) :: x 
  !f2py real(8), intent(in), dimension(num_nodes_y, dim) :: y 
  !f2py real(8), intent(in), dimension(num_nodes, num_nodes_y) :: beta
  !f2py real(8), intent(in) :: sig
  !f2py integer, intent(in) :: ord
  !f2py real(8), intent(out), dimension(num_nodes, dim) :: f

  real(8) :: ut
  real(8) :: lpt
  integer :: k,l 
  real(8) :: df(dim)
  real(8) :: c_(5, 5), c1_(4, 4)
  c_= reshape((/1.,1.,1.,1.,1.,    0.,1.,1.,1.,1.,    0.,0., 1./3, 0.4, 3./7,  &
       0.,0.,0., 1./15, 2./21,   0.,0.,0.,0., 1./105 /), (/5,5/))
  c1_ = reshape((/1., 1./3, 1./5, 1./7,    0., 1./3, 1./5, 1./7,    0.,0., 1./15, 2./35,&
       0.,0.,0., 1./105 /), (/4,4/))

  !$omp parallel do private(k,l,ut,lpt,Kh_diff,df) shared &
  !$omp& (num_nodes, num_nodes_y, f, sig, ord, beta, c_, c1_)
  do k = 1, num_nodes, 1
     df = 0
     do l = 1, num_nodes_y, 1
        ut = norm2(x(k,:) - y(l,:)) / sig
        if (ord > 4) then
           ut = ut * ut
           if (ut < 1e-8) then
              Kh_diff = - 1.0/(2*sig**2)
           else
              Kh_diff = (-1.0)* exp(-0.5*ut)/(2*sig**2)
           end if
        else           
           if (ut < 1e-8) then
              Kh_diff = - c1_(ord,1)/(2*sig*sig)
           else
              lpt = c1_(ord,1) + c1_(ord,2)*ut + c1_(ord,3)*ut**2 + c1_(ord,4)*ut**3 
              Kh_diff = -lpt * exp(-1.0*ut)/(2*sig**2)
           end if
        end if
        df = df + Kh_diff * 2*(x(k,:)-y(l,:))* beta(k,l)
     end do
     f(k, :) = df
  end do
  !$omp end parallel do
end subroutine applykdiffmat
