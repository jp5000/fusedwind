from numpy import interp, ndarray, array, loadtxt, log, zeros, cos, arccos, sin
from numpy import nonzero, argsort, NaN, isnan, mean, ones, vstack, linspace, exp
from numpy import arctan, arange, pi, sqrt, dot, hstack
from numpy.linalg.linalg import norm
from scipy.interpolate import interp1d
from scipy.integrate import quad

from openmdao.lib.datatypes.api import VarTree, Float, Instance, Slot, Array, List, Int, Str, Dict
from openmdao.main.api import Driver
# , IOInterface
from openmdao.main.api import Component, Assembly, VariableTree, Container
from openmdao.main.interfaces import implements, ICaseRecorder, ICaseIterator
from openmdao.main.case import Case

# KLD - 8/29/13 separated vt and assembly into separate file
from vt import GenericWindTurbineVT, GenericWindTurbinePowerCurveVT, \
    ExtendedWindTurbinePowerCurveVT, GenericWindFarmTurbineLayout, \
    GenericWindRoseVT

from fusedwind.interface import base, implement_base, InterfaceInstance
from fusedwind.fused_helper import *

# ------------------------------------------------------------
# Components and Assembly Base Classes

# most basic plant flow components
@base
class CDFBase(Component):
    """cumulative distribution function"""

    x = Array(iotype='in', desc='input curve')

    F = Array(iotype='out')

@base
class BaseAEPAggregator(Component):
    """
    Assumes implementing component provides overall plant energy output.
    """

    # Outputs
    gross_aep = Float(0.0, iotype='out', units='kW*h',
        desc='Gross Annual Energy Production before availability and loss impacts')
    net_aep = Float(0.0, iotype='out', units='kW*h',
        desc='Net Annual Energy Production after availability and loss impacts')


@implement_base(BaseAEPAggregator)
class BaseAEPAggregator_NoFlow(Component):
    """
    Assumes implementing component takes individual turbine output and combines with loss factors and turbine number to get plant energy output.
    """

    # parameters
    array_losses = Float(0.059, iotype='in', desc='energy losses due to turbine interactions - across entire plant')
    other_losses = Float(0.0, iotype='in', desc='energy losses due to blade soiling, electrical, etc')
    availability = Float(0.94, iotype='in', desc='average annual availbility of wind turbines at plant')
    turbine_number = Int(100, iotype='in', desc='total number of wind turbines at the plant')
    machine_rating = Float(5000.0, iotype='in', desc='machine rating of turbine')

    # Outputs
    gross_aep = Float(0.0, iotype='out', units='kW*h',
        desc='Gross Annual Energy Production before availability and loss impacts')
    net_aep = Float(0.0, iotype='out', units='kW*h',
        desc='Net Annual Energy Production after availability and loss impacts')
    capacity_factor = Float(0.0, iotype='out', desc='Capacity factor for wind plant') # ??? generic or specific? will be easy to calculate, #


#####################

# wind rose based components

class WeibullWindRose(Component):

    """Calculates the frequency_array using a weibull distribution"""
    wind_directions = List(iotype='in', units='deg',
        desc='Direction sectors angles [n_wd]')
    wind_speeds = List(iotype='in', units='m/s',
        desc='wind speeds sectors [n_ws]')
    wind_rose_array = Array([], iotype='in', units='m/s',
        desc='Windrose array [wind_directions, frequency, weibull_A, weibull_k]')
    cut_in = Float(4.0, iotype='in',
        desc='The cut-in wind speed of the wind turbine')
    cut_out = Float(25.0, iotype='in',
        desc='The cut-out wind speed of the wind turbine')

    wind_rose = VarTree(GenericWindRoseVT(), iotype='out',
        desc='A wind rose VT')

    @property
    def n_wd(self):
        """Number of wind direction bins"""
        return len(self.wind_directions)

    @property
    def n_ws(self):
        """Number of wind speed bins"""
        return len(self.wind_speeds)

    def test_consistency_inputs(self):
        """Test the consistency of the inputs. This will be optimized away in production"""
        assert len(self.wind_rose_array) > 0,\
            'wind_rose_array is empty: %r' % self.wind_rose_array
        assert len(self.wind_speeds) > 0,\
            'wind_speeds is empty: %r' % self.wind_speeds
        assert len(self.wind_directions) > 0,\
            'wind_directions is empty: %r' % self.wind_directions
        assert isinstance(self.wind_rose_array, ndarray),\
            'The wind rose array is a ndarray of 4 columns'
        assert self.wind_rose_array.shape[1] == 4,\
            'wind_rose_array = array([wind_directions, frequency, weibull_A, weibull_k])'
        assert mean(self.wind_directions) > 20.0,\
            'Wind direction should be given in degrees'
        assert self.wind_rose_array[:, 0].mean() > 20.0,\
            'The first column of wind_rose should be in degrees'
        assert 1.0 - sum(self.wind_rose_array[:, 1]) < 1.0E-3,\
            'The second column of self.wind_rose_array should sum to 1.0'

    def test_consistency_outputs(self):
        """Test the consistency of the outputs. This will be optimized away in production"""
        assert sum(self.wind_rose.frequency_array.flatten()) <= 1.0 + 1.0E-3, \
            'The frequency array should never reach 1.0, because there are some high wind speeds not considered in the array.'

    def execute(self):
        """Calculate the new wind_rose"""
        if len(self.wind_rose_array) == 0:
            return

        self.test_consistency_inputs()

        self._directions = self.wind_rose_array[:, 0]
        _direction_frequency = self.wind_rose_array[:, 1]
        self._weibull_A = self.wind_rose_array[:, 2]
        self._weibull_k = self.wind_rose_array[:, 3]

        # Creating the interpolation functions
        indis = range(self.wind_rose_array.shape[0])
        indis.extend([0])
        _d_wd = self._directions[1] - self._directions[0]
        diff_dir = _direction_frequency[indis] / _d_wd
        directions = self._directions.tolist() + [360.0]

        def corr_dir(d):
            if d < 0.0:
                return d + 360.0
            if d >= 360.0:
                return d - 360.0
            return d
        pdf_wd = interp1d(directions, diff_dir, kind='linear')
        pdf_wd_ext = lambda d: pdf_wd(corr_dir(d))
        weibull_A = interp1d(directions, self._weibull_A[indis], kind='linear')
        weibull_k = interp1d(directions, self._weibull_k[indis], kind='linear')
        self.wind_rose.wind_directions = self.wind_directions
        self.wind_rose.wind_speeds = self.wind_speeds

        # Statistical functions
        pdf_ws = lambda A, k, u: k / A * \
            (u / A) ** (k - 1.0) * exp(-(u / A) ** k)
        #cdf_ws = lambda A,k,u, bin=1.0: exp(-(max(0.0, u-bin/2.0)/A)**k) - exp(-((u+bin/2.0)/A)**k)
        cdf_ws = lambda A, k, u0, u1: exp(-(u0 / A) ** k) - exp(-(u1 / A) ** k)
        # TODO: Fine tune this epsrel
        cdf_wd = lambda wd0, wd1: quad(pdf_wd_ext, wd0, wd1, epsrel=0.1)[0]

        # Create the array
        self.wind_rose.frequency_array = zeros([self.n_wd, self.n_ws])

        for iwd, wd in enumerate(self.wind_directions):

            A = weibull_A(wd)
            k = weibull_k(wd)

            # We include all the wind directions between each wind directions
            if iwd == 0:
                wd0 = 0.5 * (self.wind_directions[-1] + wd + 360)
            else:
                wd0 = 0.5 * (self.wind_directions[iwd - 1] + wd)
            if iwd == len(self.wind_directions) - 1:
                wd1 = 0.5 * (self.wind_directions[0] + wd + 360)
            else:
                wd1 = 0.5 * (self.wind_directions[iwd + 1] + wd)

            if wd0 > wd1:  # deal with the 360-0 issue
                P_dir = cdf_wd(wd0, 359.99) + cdf_wd(0.0, wd1)
            else:
                P_dir = cdf_wd(wd0, wd1)
            # print 'between', wd0, wd1, P_dir
            for iws, ws in enumerate(self.wind_speeds):
                if iws == 0:  # We include all the cases from cut_in
                    ws0 = self.cut_in
                else:
                    ws0 = 0.5 * (self.wind_speeds[iws - 1] + ws)
                if iws == len(self.wind_speeds) - 1:
                    ws1 = self.cut_out
                else:
                    ws1 = 0.5 * (self.wind_speeds[iws + 1] + ws)
                ws0 = min(self.cut_out, max(self.cut_in, ws0))
                ws1 = min(self.cut_out, max(self.cut_in, ws1))

                self.wind_rose.frequency_array[
                    iwd, iws] = P_dir * cdf_ws(A, k, ws0, ws1)

        self.test_consistency_outputs()


@base
class GenericWindFarm(Component):

    # Inputs:
    wind_speed = Float(iotype='in', low=0.0, high=100.0, units='m/s',
        desc='Inflow wind speed at hub height')
    wind_direction = Float(iotype='in', low=0.0, high=360.0, units='deg',
        desc='Inflow wind direction at hub height')
    wt_layout = VarTree(GenericWindFarmTurbineLayout(), iotype='in',
        desc='wind turbine properties and layout')

    # Outputs:
    power = Float(iotype='out', units='kW',
        desc='Total wind farm power production')
    thrust = Float(iotype='out', units='N',
        desc='Total wind farm thrust')
    wt_power = Array([], iotype='out',
        desc='The power production of each wind turbine')
    wt_thrust = Array([], iotype='out',
        desc='The thrust of each wind turbine')


@base
class GenericWindRoseCaseGenerator(Component):

    """Component prepare all the wind speeds, directions and frequencies inputs to the AEP calculation"""
    wind_speeds = List([], iotype='in', units='m/s',
        desc='The different wind speeds to run [nWS]')
    wind_directions = List([], iotype='in', units='deg',
        desc='The different wind directions to run [nWD]')

    all_wind_speeds = List(iotype='out', units='m/s',
        desc='The different wind speeds to run [nWD*nWS]')
    all_wind_directions = List(iotype='out', units='deg',
        desc='The different wind directions to run [nWD*nWS]')
    all_frequencies = List(iotype='out',
        desc='The different wind directions to run [nWD*nWS]')


@implement_base(GenericWindRoseCaseGenerator)
class SingleWindRoseCaseGenerator(Component):

    """Component prepare all the wind speeds, directions and frequencies inputs to the AEP calculation"""
    wind_speeds = List([], iotype='in', units='m/s',
        desc='The different wind speeds to run [nWS]')
    wind_directions = List([], iotype='in', units='deg',
        desc='The different wind directions to run [nWD]')
    wind_rose = Array([], iotype='in',
        desc='Probability distribution of wind speed, wind direction [nWS, nWD]')

    all_wind_speeds = List(iotype='out', units='m/s',
        desc='The different wind speeds to run [nWD*nWS]')
    all_wind_directions = List(iotype='out', units='deg',
        desc='The different wind directions to run [nWD*nWS]')
    all_frequencies = List(iotype='out',
        desc='The different wind directions to run [nWD*nWS]')

    def execute(self):
        # Not needed anymore
        # wr = WeibullWindRose()(wind_directions=self.wind_directions, wind_speeds=self.wind_speeds,
        #                       wind_rose_array=self.wind_rose).wind_rose

        self.all_wind_directions = []
        self.all_wind_speeds = []
        self.all_frequencies = []
        if self.wind_rose.size > 0:
            for i_ws, ws in enumerate(self.wind_speeds):
                for i_wd, wd in enumerate(self.wind_directions):
                    self.all_wind_directions.append(wd)
                    self.all_wind_speeds.append(ws)
                    self.all_frequencies.append(self.wind_rose[i_wd, i_ws])
        else:
            print self.__class__.__name__, 'input, wind_rose is empty'


@implement_base(GenericWindRoseCaseGenerator)
class MultipleWindRosesCaseGenerator(Component):

    """Component prepare all the wind speeds, directions and frequencies inputs to the AEP calculation.
    """

    wind_speeds = List([], iotype='in', units='m/s',
        desc='The different wind speeds to run [nWS]')
    wind_directions = List([], iotype='in', units='deg',
        desc='The different wind directions to run [nWD]')
    wt_layout = VarTree(GenericWindFarmTurbineLayout(), iotype='in',
        desc='the wind farm layout')

    all_wind_speeds = List(iotype='out', units='m/s',
        desc='The different wind speeds to run [nWD*nWS]')
    all_wind_directions = List(iotype='out', units='deg',
        desc='The different wind directions to run [nWD*nWS]')
    all_frequencies = List(iotype='out',
        desc='The different wind directions to run [nWD*nWS][nWT]')

    def execute(self):
        self.all_wind_directions = []
        self.all_wind_speeds = []
        self.all_frequencies = []
        for wt in self.wt_layout.wt_list:
            wt.wind_rose.change_resolution(wind_directions=self.wind_directions, wind_speeds=self.wind_speeds)
        for i_ws, ws in enumerate(self.wind_speeds):
            for i_wd, wd in enumerate(self.wind_directions):
                self.all_wind_directions.append(wd)
                self.all_wind_speeds.append(ws)
                self.all_frequencies.append([wt.wind_rose.frequency_array[i_wd, i_ws] for wt in self.wt_layout.wt_list])


@base
class GenericPostProcessWindRose(Component):

    """Using the same wind rose for all the wind turbines"""
    # Inputs
    wind_speeds = List([], iotype='in', units='m/s',
        desc='The different wind speeds to run [nWS]')
    wind_directions = List([], iotype='in', units='deg',
        desc='The different wind directions to run [nWD]')
    frequencies = List([], iotype='in',
        desc='The different wind directions to run [nWD*nWS]')
    powers = List([], iotype='in', units='kW*h',
        desc='The different wind directions to run [nWD*nWS]')

    # Outputs
    net_aep = Float(0.0, iotype='out', units='kW*h',
        desc='Annual Energy Production')
    gross_aep = Float(0.0, iotype='out', units='kW*h',
        desc='Gross Annual Energy Production')
    capacity_factor = Float(0.0, iotype='out',
        desc='Capacity factor')
    array_aep = Array([], iotype='out', units='kW*h',
        desc='The energy production per sector [nWD, nWS]')


@implement_base(GenericPostProcessWindRose)
class PostProcessSingleWindRose(Component):

    """Using the same wind rose for all the wind turbines"""
    # Inputs
    wind_speeds = List([], iotype='in', units='m/s',
        desc='The different wind speeds to run [nWS]')
    wind_directions = List([], iotype='in', units='deg',
        desc='The different wind directions to run [nWD]')
    frequencies = List([], iotype='in',
        desc='The different wind directions to run [nWD*nWS]')
    powers = List([], iotype='in', units='kW*h',
        desc='The different wind directions to run [nWD*nWS]')

    # Outputs
    net_aep = Float(0.0, iotype='out', units='kW*h',
        desc='Annual Energy Production')
    gross_aep = Float(0.0, iotype='out', units='kW*h',
        desc='Gross Annual Energy Production')
    capacity_factor = Float(0.0, iotype='out',
        desc='Capacity factor')
    array_aep = Array([], iotype='out', units='kW*h',
        desc='The energy production per sector [nWD, nWS]')

    def execute(self):
        list_aep = [freq * power * 24 * 365 for freq,
                    power in zip(self.frequencies, self.powers)]
        self.net_aep = sum(list_aep)
        # TODO: FIX gross_aep and capacity factor
        #self.gross_aep = self.net_aep
        #self.capacity_factor = self.net_aep / self.gross_aep

        if len(self.wind_speeds) > 0 and len(self.wind_directions) > 0:
            self.array_aep = array(list_aep).reshape([len(self.wind_speeds), len(self.wind_directions)])
        else:
            print self.__class__.__name__, 'inputs, wind_speed or wind_directions are empty'


@implement_base(GenericPostProcessWindRose)
class PostProcessMultipleWindRoses(Component):

    """Use a different wind rose for each wind turbine"""
    # Inputs
    wind_speeds = List([], iotype='in', units='m/s',
        desc='The different wind speeds to run [nWS]')
    wind_directions = List([], iotype='in', units='deg',
        desc='The different wind directions to run [nWD]')
    frequencies = List([], iotype='in',
        desc='The different wind directions to run [nWD*nWS][nWT]')
    powers = List([], iotype='in', units='kW*h',
        desc='The different wind directions to run [nWD*nWS][nWT]')

    # Outputs
    net_aep = Float(0.0, iotype='out', units='kW*h',
        desc='Net Annual Energy Production')
    gross_aep = Float(0.0, iotype='out', units='kW*h',
        desc='Gross Annual Energy Production')
    capacity_factor = Float(0.0, iotype='out',
        desc='Capacity factor')
    array_aep = Array([], iotype='out', units='kW*h',
        desc='The energy production per sector [nWD, nWS]')
    wt_aep = Array([], iotype='out', units='kW*h',
        desc='The energy production per turbine [nWT]')


    def execute(self):
        nwd, nws = len(self.wind_directions), len(self.wind_speeds)
        assert len(self.frequencies) == nws * nwd
        assert len(self.powers) == nws * nwd

        array_aep = array([array(freq) * array(power) * 24 *
                           365 for freq, power in zip(self.frequencies, self.powers)])
        self.net_aep = array_aep.sum()
        # TODO: FIX gross_aep and capacity factor
        #self.gross_aep = array([array(freq) * array(power).max() * 24 * 365 for freq, power in zip(self.frequencies, self.powers)]).sum()
        #self.capacity_factor = self.net_aep / self.gross_aep

        self.array_aep = array_aep.sum(1).reshape([len(self.wind_directions), len(self.wind_speeds)])
        self.wt_aep = array_aep.sum(0)



### TODO: Move these components to FUSED-Wake ##########################################################################


@base
class GenericWSPosition(Component):

    """Calculate the positions where we should calculate the wind speed on the rotor"""
    wt_desc = VarTree(GenericWindTurbineVT(), iotype='in')
    ws_positions = Array([], iotype='out', units='m',
        desc='the position [n,3] of the ws_array')
    wt_xy = List([0.0, 0.0], iotype='in', units='m',
        desc='The x,y position of the wind turbine')


@implement_base(GenericWSPosition)
class HubCenterWSPosition(Component):

    """
    Generate the positions at the center of the wind turbine rotor
    """
    wt_desc = VarTree(GenericWindTurbineVT(), iotype='in')
    ws_positions = Array([], iotype='out', units='m',
        desc='the position [n,3] of the ws_array')
    wt_xy = List([0.0, 0.0], iotype='in', units='m',
        desc='The x,y position of the wind turbine')

    def execute(self):
        self.ws_positions = array([[self.wt_xy[0], self.wt_xy[1], self.wt_desc.hub_height]])


@base
class GenericWakeSum(Component):

    """
    Generic class for calculating the wake accumulation
    """
    wakes = List(iotype='in',
        desc='wake contributions to rotor wind speed [nwake][n]')
    ws_array_inflow = Array(iotype='in', units='m/s',
        desc='inflow contributions to rotor wind speed [n]')

    ws_array = Array(iotype='out', units='m/s',
        desc='the rotor wind speed [n]')

@base
class GenericHubWindSpeed(Component):

    """
    Generic class for calculating the wind turbine hub wind speed.
    Typically used as an input to a wind turbine power curve / thrust coefficient curve.
    """
    ws_array = Array([], iotype='in', units='m/s',
        desc='an array of wind speed on the rotor')

    hub_wind_speed = Float(0.0, iotype='out', units='m/s',
        desc='hub wind speed')


@base
class GenericFlowModel(Component):

    """
    Framework for a flow model
    """
    ws_positions = Array([], iotype='in',units='m',
        desc='the positions of the wind speeds in the global frame of reference [n,3] (x,y,z)')
    ws_array = Array([], iotype='out', units='m/s',
        desc='array of wind speed at ws_positions')


@implement_base(GenericFlowModel)
class GenericWakeModel(Component):

    """
    Framework for a wake model
    """
    # Inputs
    wt_desc = VarTree(GenericWindTurbineVT(), iotype='in',
        desc='the geometrical description of the current turbine')
    ws_positions = Array([], iotype='in', units='m',
        desc='the positions of the wind speeds in the global frame of reference [n,3] (x,y,z)')
    wt_xy = List([0.0, 0.0], iotype='in', units='m',
        desc='The x,y position of the current wind turbine')
    c_t = Float(0.0, iotype='in',
        desc='the thrust coefficient of the wind turbine')
    ws_array_inflow = Array([], iotype='in', units='m/s',
        desc='The inflow velocity at the ws_positions')
    wind_direction = Float(0.0, iotype='in', units='deg',
        desc='The inflow wind direction')
    # Outputs
    ws_array = Array([], iotype='out', units='m/s',
        desc='array of wind speed at ws_positions')
    du_array = Array([], iotype='out', units='m/s',
        desc='The deficit in m/s. Empty if only zeros')


@implement_base(GenericFlowModel)
class GenericInflowGenerator(Component):

    """
    Framework for an inflow model
    """
    wind_speed = Float(0.0, iotype='in', units='m/s',
        desc='the reference wind speed')
    ws_positions = Array([], iotype='in',units='m',
        desc='the positions of the wind speeds in the global frame of reference [n,3] (x,y,z)')
    ws_array = Array([], iotype='out', units='m/s',
        desc='array of wind speed at ws_positions')

@implement_base(GenericInflowGenerator)
class PowerLawInflowGenerator(Component):

    """
    Framework for an inflow Power law inflow flow model
    """
    # Inputs
    wind_speed = Float(0.0, iotype='in', units='m/s',
        desc='the reference wind speed')
    ws_positions = Array([], iotype='in',units='m',
        desc='the positions of the wind speeds in the global frame of reference [n,3] (x,y,z)')
    z_ref = Float(100., iotype='in',units='m',
        desc='the reference height above ground level')
    shear_coef =  Float(0.11, iotype='in',
        desc='vertical wind speed profile shear coefficient')

    # Outputs
    ws_array = Array([], iotype='out', units='m/s',
        desc='array of wind speed at ws_positions')

    def execute(self):
        self.ws_array = (self.wind_speed)*((self.ws_positions[:,2]/self.z_ref)** \
                        self.shear_coef)

@implement_base(GenericInflowGenerator)
class LogLawInflowGenerator(Component):

    """
    Framework for an inflow Log law inflow flow model
    """
    # Inputs
    wind_speed = Float(0.0, iotype='in', units='m/s',
        desc='the reference wind speed')
    z_ref = Float(100., iotype='in',units='m',
        desc='the reference height above ground level')
    ws_positions = Array([], iotype='in',units='m',
        desc='the positions of the wind speeds in the global frame of reference [n,3] (x,y,z)')
    z_0 = Float(0.0002, iotype='in',units='m',
        desc='the surface roughness length')
    displacement_0 = Float(0., iotype='in',units='m',
        desc='the zero height displacement length')
    L = Float(NaN, iotype='in',units='m',
        desc='the Monin-Obukhov stability parameter length')
    z_i = Float(400., iotype='in',units='m',
        desc='boundary layer height')
    stab_term = Int(0, iotype='in',units='m',
        desc='the universal stability function (psi) selector')
    # Outputs
    ws_array = Array([], iotype='out', units='m/s',
        desc='array of wind speed at ws_positions')

    def execute(self):
        ws_ref = self.wind_speed
        z = self.ws_positions[:,2]
        d = self.displacement_0
        z_ref = self.z_ref
        L = self.L
        z_0 = self.z_0

        if  isnan(L): # Neutral case
            self.ws_array = ws_ref*log((z-d)/z_0)/log((z_ref-d)/z_0)

        elif self.stab_term == 0:
            '''
            Pena formulation:

            A. Pena, T. Mikkelsen, S.-E. Gryning, C.B. Hasager, A.N. Hahmann,
            M. Badger, et al.,
            Offshore vertical wind shear, DTU Wind Energy-E-Report-0005(EN),
            Technical University of Denmark, 2012.
            '''
            aux = L/max(z)
            #Test the consistency of the inputs.
            assert (1.-12.*max(z)/L) >= 0,\
            'Too high Monin-Obukhov Length. L must be larger than 12*max(z): '+ repr(aux)

            x = (1.-12.*z/L)**(1./3.)
            psi_m = (z/L>=0)*(-4.7*z/L)+(z/L<0)*(3./2.*log((1+x+x**2.)/3.) - \
                    sqrt(3.)*arctan((2.*x+1.)/sqrt(3.)) + pi/sqrt(3.) )

            x_ref = (1.-12.*z_ref/L)**(1./3.)
            psi_m_ref = (z_ref/L>=0)*(-4.7*z_ref/L)+\
                        (z_ref/L<0)*(3./2.*log((1+x_ref+x_ref**2.)/3.) - \
                        sqrt(3.)*arctan((2.*x_ref+1.)/sqrt(3.)) + pi/sqrt(3.))

            self.ws_array = ws_ref*(log((z-d)/z_0)    - psi_m)/ \
                                   (log((z_ref-d)/z_0)- psi_m_ref)

        elif self.stab_term == 1:
            '''
            Businger-Dyer formulation:

            J. A. Businger, J. C. Wyngaard, Y. Izumi, and E. F. Bradley, 1971:
            Flux-Profile Relationships in the Atmospheric Surface Layer.
            J. Atmos. Sci., 28, 181-189
            '''
            kappa = 0.4  # Von Karman Constant
            gamma = 19.3 # Empirical parameter derived from Kansas measurements
            beta  = 4.8  # Empirical parameter derived from Kansas measurements

            aux = L/max(z)
            #Test the consistency of the inputs.
            assert (1.-gamma*max(z)/L) >= 0,\
            'Too high Monin-Obukhov Length. L must be larger than '+repr(gamma)+'*max(z): '+ repr(aux)

            phi_m = (z/L<0) * (1.-gamma*z/L)**0.25
            psi_m = (z/L<0)*(2.*log((1.+phi_m**2.)/2.)-2.*arctan(phi_m)+pi/2.) + \
                    (z/L>=0)*(-beta*z/L)

            phi_m_ref = (z_ref/L<0) * (1.-gamma*z_ref/L)**0.25
            psi_m_ref = (z_ref/L<0)*(2.*log((1.+phi_m_ref**2.)/2.) - \
                         2.*arctan(phi_m_ref) + pi/2. ) + \
                         (z_ref/L>=0)*(-beta*z_ref/L)


            self.ws_array = ws_ref*(log((z-d)/z_0)    - psi_m)/ \
                                   (log((z_ref-d)/z_0)- psi_m_ref)


@base
class GenericWindTurbine(Component):
    hub_wind_speed = Float(iotype='in')

    power = Float(0.0, iotype='out', units='W',
        desc='The wind turbine power')
    thrust = Float(
        0.0, iotype='out', units='N',
        desc='The wind turbine thrust')
    c_t = Float(0.0, iotype='out',
        desc='The wind turbine thrust coefficient')


@implement_base(GenericWindTurbine)
class WindTurbinePowerCurve(Component):

    """
    wt_desc needs to contain:
        - power_curve
        - c_t_curve
        - rotor_diameter
    """
    wt_desc = VarTree(GenericWindTurbinePowerCurveVT(), iotype='in',
        desc='The wind turbine description')
    hub_wind_speed = Float(0.0, iotype='in',
        desc='Wind Speed at hub height')
    density = Float(1.225, iotype='in',
        desc='Air density')

    power = Float(0.0, iotype='out',
        desc='The wind turbine power')
    thrust = Float(0.0, iotype='out',
        desc='The wind turbine thrust')
    c_t = Float(0.0, iotype='out',
        desc='The wind turbine thrust coefficient')
    a = Float(0.0, iotype='out',
        desc='The wind turbine induction factor')

    def execute(self):
        self.power = interp(self.hub_wind_speed, self.wt_desc.power_curve[:, 0], self.wt_desc.power_curve[:, 1])
        self.c_t = min(interp(self.hub_wind_speed, self.wt_desc.c_t_curve[:, 0], self.wt_desc.c_t_curve[:, 1]), 1.0)

        if self.hub_wind_speed < self.wt_desc.c_t_curve[:, 0].min():
            self.power = 0.0
            self.c_t = 0.0
        self._set_a()
        self._set_thrust()

    def _set_a(self):
        """
        Set the induced velocity based on the thrust coefficient
        """
        self.a = 0.5 * (1.0 - sqrt(1.0 - self.c_t))

    def _set_thrust(self):
        """
        Set the thrust based on the thrust coefficient
        """
        self.thrust = self.c_t * self.density * self.hub_wind_speed ** 2.0 * \
            self.wt_desc.rotor_diameter ** 2.0 * pi / 4.0


# if __name__ == '__main__':
    # in_pl = PowerLawInflowGenerator()
    # in_pl.wind_speed = 10.
    # in_pl.z_ref = 100.
    # in_pl.ws_positions = array([[0.,0.,50.],[0.,0.,100.],[0.,0.,1000.]])
    # in_pl.shear_coef = 0.1
    # in_pl.run()
    # print in_pl.ws_array
    # assert (in_pl.ws_array==array([0.,1.,10.**1.1])).all

    # in_log = LogLawInflowGenerator()
    # in_log.wind_speed = 10.
    # in_log.z_ref = 100.
    # in_log.ws_positions = array([[0.,0.,50.],[0.,0.,100.],[0.,0.,1000.]])
    # in_log.z_0 = 0.0002
    # in_log.L = NaN
    # in_log.run()
    # print in_log.ws_array


    # in_log2 = LogLawInflowGenerator()
    # in_log2.wind_speed = 10.
    # in_log2.z_ref = 100.
    # in_log2.ws_positions = array([[0.,0.,50.],[0.,0.,100.],[0.,0.,1000.]])
    # in_log2.z_0 = 0.0002
    # in_log2.L = 20000
    # in_log2.stab_term = 0
    # in_log2.run()
    # print in_log2.ws_array

    # in_log2 = LogLawInflowGenerator()
    # in_log2.wind_speed = 10.
    # in_log2.z_ref = 100.
    # in_log2.ws_positions = array([[0.,0.,50.],[0.,0.,100.],[0.,0.,1000.]])
    # in_log2.z_0 = 0.0002
    # in_log2.L = -1000
    # in_log2.stab_term = 0
    # in_log2.run()
    # print in_log2.ws_array

    # in_log3 = LogLawInflowGenerator()
    # in_log3.wind_speed = 10.
    # in_log3.z_ref = 100.
    # in_log3.ws_positions = array([[0.,0.,50.],[0.,0.,100.],[0.,0.,1000.]])
    # in_log3.z_0 = 0.0002
    # in_log3.L = 20000
    # in_log3.stab_term = 1
    # in_log3.run()
    # print in_log3.ws_array

    # in_log3 = LogLawInflowGenerator()
    # in_log3.wind_speed = 10.
    # in_log3.z_ref = 100.
    # in_log3.ws_positions = array([[0.,0.,50.],[0.,0.,100.],[0.,0.,1000.]])
    # in_log3.z_0 = 0.0002
    # in_log3.L = -1000
    # in_log3.stab_term = 1
    # in_log3.run()
    # print in_log3.ws_array