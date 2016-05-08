// ###################################################################
// #### This file is part of the artificial intelligence project, and is
// #### offered under the licence agreement described on
// #### http://www.mrsoft.org/
// ####
// #### Copyright:(c) 2014, Michael R. . All rights reserved.
// ####
// #### Unless required by applicable law or agreed to in writing, software
// #### distributed under the License is distributed on an "AS IS" BASIS,
// #### WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// #### See the License for the specific language governing permissions and
// #### limitations under the License.
// ###################################################################

unit Haar2DAdaBoost;

// ############################################################
// #### Ada boost learner with post optimization
// #### according to Lienhardt et. al.: An etended set of Haar-like features for rapid object detection
// ############################################################

interface

uses AdaBoost, BaseMathPersistence, BaseClassifier, Types, CustomBooster, integralImg,
     Haar2D;

type
  THaar2DBoostProps = record
    baseProps : TBoostProperties;

    winWidth : integer;
    winHeight : integer;
    numColorPlanes : integer;
    FeatureTypes : TIntegralType;
  end;

type
  THaar2DProps = class(TBaseMathPersistence)
  private
    fWinWidth, fWinHeight : integer;
    fNumColorPlanes : integer;
    fFeatureType : TIntegralType;
    fRefFeatureList : THaar2DRefFeatureList;
  protected
    procedure OnLoadIntProperty(const Name : String; Value : integer); override;
    class function ClassIdentifier : String; override;
    procedure DefineProps; override;
    function PropTypeOfName(const Name : string) : TPropType; override;
    procedure FinishReading; override;
  public
    property RefFeatureList : THaar2DRefFeatureList read fRefFeatureList;

    function WinWidth : integer;
    function WinHeight : integer;
    function FeatureType : TIntegralType;
    function NumColorPlanes : integer;

    constructor Create(aWinWidth, aWinHeight : integer; aNumColPlanes : integer; aFeatureType : TIntegralType);
    destructor Destroy; override;
  end;

// ############################################################
// #### Extension to the standard ada boost classifier (added the classification offset)
type
  THaar2DAdaBoost = class(TDiscreteAdaBoostClassifier)
  private
    function GetAddObj: THaar2DProps;
  public
    property Props : THaar2DProps read GetAddObj;

    procedure SetProperties(aWinWidth, aWinHeight : integer; aNumColPlanes : integer; aFeatureType : TIntegralType);
  end;

// ####################################################
// #### Base adaboost algorithm
type
  THaar2DDiscreteAdaBoostLearner = class(TDiscreteAdaBoostLearner)
  private
    fwinWidth : integer;
    fwinHeight : integer;
    fnumColorPlanes : integer;
    fFeatureTypes : TIntegralType;
  protected
    function DoLearn(const weights : Array of double) : TCustomClassifier; override;
  public
    procedure SetProperties(const props : THaar2DBoostProps);

    class function CanLearnClassifier(Classifier : TCustomClassifierClass) : boolean; override;
    class function BoostClass : TCustomBoostingClassifierClass; override;
  end;

// ############################################################
// #### Extension to the standard gentle boost classifier (added the classification offset)
type
  THaar2DGentleBoost = class(TGentleBoostClassifier)
  private
    function GetAddObj: THaar2DProps;
  public
    property Props : THaar2DProps read GetAddObj;

    procedure SetProperties(aWinWidth, aWinHeight : integer; aNumColPlanes : integer; aFeatureType : TIntegralType);
  end;

// ####################################################
// #### Base adaboost algorithm
type
  THaar2DGentleBoostLearner = class(TGentleBoostLearner)
  private
    fwinWidth : integer;
    fwinHeight : integer;
    fnumColorPlanes : integer;
    fFeatureTypes : TIntegralType;
  protected
    function DoLearn(const weights : Array of double) : TCustomClassifier; override;
  public
    procedure SetProperties(const props : THaar2DBoostProps);

    class function CanLearnClassifier(Classifier : TCustomClassifierClass) : boolean; override;
    class function BoostClass : TCustomBoostingClassifierClass; override;
  end;


implementation

uses SysUtils, SimpleDecisionStump;

const cHaar2DPropWinWidth = 'haar2DWinWidth';
      cHaar2DPropWinHeight = 'haar2DWinHeight';
      cHaar2DPropNumColPlane = 'haar2DNumColorPlane';
      cHaar2DPropFeatureType = 'haar2DFeatureType';

{ TDiscreteAdaBoostLearnerEx }

class function THaar2DDiscreteAdaBoostLearner.BoostClass: TCustomBoostingClassifierClass;
begin
     Result := THaar2DAdaBoost;
end;

class function THaar2DDiscreteAdaBoostLearner.CanLearnClassifier(
  Classifier: TCustomClassifierClass): boolean;
begin
     Result := Classifier = THaar2DAdaBoost;
end;

function THaar2DDiscreteAdaBoostLearner.DoLearn(
  const weights: array of double): TCustomClassifier;
begin
     Result := inherited DoLearn(weights);

     (Result as THaar2DAdaBoost).SetProperties(fwinWidth, fwinHeight, fnumColorPlanes, fFeatureTypes);
end;

procedure THaar2DDiscreteAdaBoostLearner.SetProperties(
  const props: THaar2DBoostProps);
begin
     inherited SetProperties(props.baseProps);

     fwinWidth := props.winWidth;
     fwinHeight := props.winHeight;
     fnumColorPlanes := props.numColorPlanes;
     fFeatureTypes := props.FeatureTypes;
end;

{ THaar2DAdaBoost }

function THaar2DAdaBoost.GetAddObj: THaar2DProps;
begin
     Result := AddObj as THaar2DProps;
end;

procedure THaar2DAdaBoost.SetProperties(aWinWidth, aWinHeight,
  aNumColPlanes: integer; aFeatureType: TIntegralType);
begin
     SetAddObj(THaar2DProps.Create(aWinWidth, aWinHeight, aNumColPlanes, aFeatureType));
end;

{ THaar2DProps }

class function THaar2DProps.ClassIdentifier: String;
begin
     Result := 'Haar2DProps';
end;

constructor THaar2DProps.Create(aWinWidth, aWinHeight, aNumColPlanes: integer;
  aFeatureType: TIntegralType);
begin
     fWinWidth := aWinWidth;
     fWinHeight := aWinHeight;
     fNumColorPlanes := aNumColPlanes;
     fFeatureType := aFeatureType;

     fRefFeatureList := THaar2DRefFeatureList.Create(fWinWidth, fWinHeight, fNumColorPlanes, fFeatureType);
end;

procedure THaar2DProps.DefineProps;
begin
     inherited;

     AddIntProperty(cHaar2DPropWinWidth, fWinWidth);
     AddIntProperty(cHaar2DPropWinHeight, fWinHeight);
     AddIntProperty(cHaar2DPropNumColPlane, fNumColorPlanes);
     AddIntProperty(cHaar2DPropFeatureType, Integer(fFeatureType));
end;

function THaar2DProps.PropTypeOfName(const Name: string): TPropType;
begin
     if (CompareText(Name, cHaar2DPropWinWidth) = 0) or
        (CompareText(Name, cHaar2DPropWinHeight) = 0) or
        (CompareText(Name, cHaar2DPropNumColPlane) = 0) or
        (CompareText(Name, cHaar2DPropFeatureType) = 0)
     then
         Result := ptInteger
     else
         Result := inherited PropTypeOfName(Name);
end;


destructor THaar2DProps.Destroy;
begin
     fRefFeatureList.Free;

     inherited;
end;

function THaar2DProps.FeatureType: TIntegralType;
begin
     Result := fFeatureType;
end;

procedure THaar2DProps.FinishReading;
begin
     inherited;

     fRefFeatureList.Free;
     fRefFeatureList := THaar2DRefFeatureList.Create(fWinWidth, fWinHeight, fNumColorPlanes, fFeatureType);
end;

function THaar2DProps.NumColorPlanes: integer;
begin
     Result := fNumColorPlanes;
end;

procedure THaar2DProps.OnLoadIntProperty(const Name: String; Value: integer);
begin
     if CompareText(Name, cHaar2DPropWinWidth) = 0
     then
         fWinWidth := Value
     else if CompareText(Name, cHaar2DPropWinHeight) = 0
     then
         fWinHeight := Value
     else if CompareText(Name, cHaar2DPropNumColPlane) = 0
     then
         fNumColorPlanes := Value
     else if CompareText(Name, cHaar2DPropFeatureType) = 0
     then
         fFeatureType := TIntegralType(Value)
     else
         inherited;
end;

function THaar2DProps.WinHeight: integer;
begin
     Result := fWinHeight;
end;

function THaar2DProps.WinWidth: integer;
begin
     Result := fWinWidth;
end;

{ THaar2DGentleBoost }

function THaar2DGentleBoost.GetAddObj: THaar2DProps;
begin
     Result := AddObj as THaar2DProps;
end;

procedure THaar2DGentleBoost.SetProperties(aWinWidth, aWinHeight,
  aNumColPlanes: integer; aFeatureType: TIntegralType);
begin
     SetAddObj(THaar2DProps.Create(aWinWidth, aWinHeight, aNumColPlanes, aFeatureType));
end;

{ TGentleBoostLearnerEx }

class function THaar2DGentleBoostLearner.BoostClass: TCustomBoostingClassifierClass;
begin
     Result := THaar2DGentleBoost;
end;

class function THaar2DGentleBoostLearner.CanLearnClassifier(
  Classifier: TCustomClassifierClass): boolean;
begin
     Result := Classifier = THaar2DGentleBoost;
end;

function THaar2DGentleBoostLearner.DoLearn(
  const weights: array of double): TCustomClassifier;
begin
     Result := inherited DoLearn(weights);

     (Result as THaar2DGentleBoost).SetProperties(fwinWidth, fwinHeight, fnumColorPlanes, fFeatureTypes);
end;

procedure THaar2DGentleBoostLearner.SetProperties(const props: THaar2DBoostProps);
begin
     inherited SetProperties(props.baseProps);

     fwinWidth := props.winWidth;
     fwinHeight := props.winHeight;
     fnumColorPlanes := props.numColorPlanes;
     fFeatureTypes := props.FeatureTypes;
end;

initialization
  RegisterMathIO(THaar2DProps);
  RegisterMathIO(THaar2DAdaBoost);
  RegisterMathIO(THaar2DGentleBoost);

end.
