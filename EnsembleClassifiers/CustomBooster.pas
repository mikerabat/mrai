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

unit CustomBooster;

// #########################################################
// #### Common boosting properties
// #########################################################

interface

uses SysUtils, Classes, BaseMathPersistence, BaseClassifier, EnsembleClassifier, Types;

type
  TCustomBoostingClassifier = class(TEnsembelClassifier)
  private
    fAddObj : TBaseMathPersistence;
  protected
    fWeights : Array of double;
    fB : double;

    procedure SetAddObj(obj : TBaseMathPersistence);
  public
    procedure DefineProps; override;
    function PropTypeOfName(const Name : string) : TPropType; override;

    procedure OnLoadBinaryProperty(const Name : String; const Value; size : integer); override;
    procedure OnLoadDoubleProperty(const Name : string; const Value : double); override;
    function OnLoadObject(const Name : String; Obj : TBaseMathPersistence) : boolean; override;

    property AddObj : TBaseMathPersistence read fAddObj;

    constructor Create(classifierSet : TClassifierSet; ownsSet : boolean; const ClassifierWeights : Array of double; const B : double = 0); virtual;
    destructor Destroy; override;
  end;
  TCustomBoostingClassifierClass = class of TCustomBoostingClassifier;

type
  TBoostProperties = record
    NumRounds : integer;
    PruneToLowestError : boolean;
    InitClassSpecificWeights : boolean;
    OwnsLearner : boolean;
    WeakLearner : TCustomWeightedLearner;
  end;

// ####################################################
// #### Base boosting
type
  TCustomBoostLearner = class(TCustomWeightedLearner)
  private
    fProps : TBoostProperties;
    fClassifierWeights : TDoubleDynArray;
  protected
    property Props : TBoostProperties read fProps;

    function BoostRound(var weakClassifier : TCustomClassifier; var Weights : TDoubleDynArray; var classifierWeight : double) : boolean; virtual; abstract;
    function DoLearn(const weights : Array of double) : TCustomClassifier; override;
    procedure PruneToLowestTrainError(Classifiers : TClassifierSet; const weights : Array of double); virtual; abstract; 

    class function BoostClass : TCustomBoostingClassifierClass; virtual; abstract;
  public
    procedure SetProperties(const props : TBoostProperties);

    destructor Destroy; override;
  end;
  TCustomBoostLearnerClass = class of TCustomBoostLearner;

implementation

uses Math;

const cBoostWeightsProp = 'boostweights';
      cBoostOffset = 'boostOffset';
      cBoostAddObj = 'boostAddObj';

destructor TCustomBoostLearner.Destroy;
begin
     if fProps.OwnsLearner and Assigned(fprops.WeakLearner) then
        fProps.WeakLearner.Free;
     
     inherited;
end;

function TCustomBoostLearner.DoLearn(
  const weights: array of double): TCustomClassifier;
var curWeights : TDoubleDynArray;
    weightsSum : double;
    k, i : integer;
    classVal : integer;
    numClassMembers : integer;
    classifier : TClassifierSet;
    cl : TCustomClassifier;
begin
     Result := nil;
     
     classifier := TClassifierSet.Create;
     try
        // copy initial wheights
        SetLength(curWeights, High(weights) + 1);
        Move(weights[0], curWeights[0], sizeof(weights));
        SetLength(fClassifierWeights, fProps.NumRounds);

        // initialize the weak learner
        fProps.WeakLearner.Init(DataSet);

        // some papers suggest to initialize the wheights class specific
        // differences in:
        // Freund, Schapire: A Short Introduction to Boosting
        // Viola, Jones: Robust real-time object detection
        if fProps.InitClassSpecificWeights then
        begin
             classVal := DataSet[0].ClassVal;
             numClassMembers := 1;

             for k := 1 to DataSet.Count - 1 do
             begin
                  if DataSet[k].ClassVal = classVal then
                     inc(numClassMembers);
             end;

             for k := 0 to DataSet.Count - 1 do
             begin
                  if DataSet[k].ClassVal = classVal
                  then
                      curWeights[k] := curWeights[k]*1/(2*numClassMembers)
                  else
                      curWeights[k] := curWeights[k]*1/(2*(DataSet.Count - numClassMembers));
             end;
        end;

        // ###########################################################
        // #### Create the AdaBoost ensemble classifier
        for k := 0 to fProps.NumRounds - 1 do
        begin
             // normalize wheights such they result in a distribution:
             weightsSum := 0;
             for i := 0 to Length(curWeights) - 1 do
                 weightsSum := weightsSum + curWeights[i];
             weightsSum := 1/weightsSum;
             for i := 0 to Length(curWeights) - 1 do
                 curWeights[i] := curWeights[i]*weightsSum;

             // ############################################################
             // #### Create a new classifier according to the current weighting
             if not BoostRound(cl, curWeights, fClassifierWeights[k]) then
             begin
                  if Assigned(cl) then
                     classifier.AddClassifier(cl);

                  break;
             end
             else
                 classifier.AddClassifier(cl);

             DoProgress(100*k div fProps.NumRounds);
        end;

        // #############################################################
        // #### now it's time to create the final strong classifier
        // the sum of the classifier weights shall be 1
        weightsSum := 0;
        for i := 0 to classifier.Count - 1 do
            weightsSum := weightsSum + fClassifierWeights[i];
        for i := 0 to classifier.Count - 1 do
            fClassifierWeights[i] := fClassifierWeights[i]/weightsSum;

        // note sometimes there is a point where the training error
        // can raise again - e.g. the weak learner always learns the same
        // classification rule - it's then better to search for the lowest training
        // error in all rounds and remove the redundant ones.
        if fProps.PruneToLowestError then
           PruneToLowestTrainError(classifier, fClassifierWeights);

        // #############################################################
        // ##### Create final classifier
        if classifier.Count > 0
        then
            Result := BoostClass.Create(classifier, True, fClassifierWeights)
        else
            classifier.Free;
     except
           classifier.Free;

           raise;
     end;
end;

{ TCustomBoostingClassifier }

constructor TCustomBoostingClassifier.Create(classifierSet: TClassifierSet;
  ownsSet: boolean; const ClassifierWeights: array of double; const B : double);
begin
     inherited Create(classifierSet, ownsSet);

     fB := B;
     SetLength(fWeights, classifierSet.Count);
     Move(ClassifierWeights[0], fWeights[0], sizeof(double)*classifierSet.Count);
end;

procedure TCustomBoostLearner.SetProperties(const props: TBoostProperties);
begin
     fProps := props;
end;

procedure TCustomBoostingClassifier.DefineProps;
begin
     if Length(fWeights) > 0 then
        AddBinaryProperty(cBoostWeightsProp, fWeights[0], Length(fWeights)*sizeof(double));

     AddDoubleProperty(cBoostOffset, fB);
     if Assigned(fAddObj) then
        AddObject(cBoostAddObj, fAddObj);

     // save classifier properties
     inherited;
end;

function TCustomBoostingClassifier.PropTypeOfName(
  const Name: string): TPropType;
begin
     if CompareText(Name, cBoostWeightsProp) = 0
     then
         Result := ptBinary
     else if CompareText(Name, cBoostOffset) = 0
     then
         Result := ptDouble
     else if CompareText(Name, cBoostAddObj) = 0
     then
         Result := ptObject
     else
         Result := inherited PropTypeOfName(Name);
end;


destructor TCustomBoostingClassifier.Destroy;
begin
     fAddObj.Free;

     inherited;
end;

procedure TCustomBoostingClassifier.OnLoadBinaryProperty(const Name: String;
  const Value; size: integer);
begin
     if CompareText(cBoostWeightsProp, Name) = 0 then
     begin
          assert(size mod sizeof(double) = 0, 'Error size differs from double double array');
          SetLength(fWeights, size div sizeof(double));
          Move(Value, fWeights[0], size);
     end
     else
         inherited;
end;

procedure TCustomBoostingClassifier.OnLoadDoubleProperty(const Name: string;
  const Value: double);
begin
     if Name = cBoostOffset
     then
         fB := Value
     else
         inherited;
end;

function TCustomBoostingClassifier.OnLoadObject(const Name: String;
  Obj: TBaseMathPersistence): boolean;
begin
     Result := CompareText(Name, cBoostAddObj) = 0;

     if Result
     then
         SetAddObj(obj)
     else
         Result := inherited OnLoadObject(Name, Obj);
end;

procedure TCustomBoostingClassifier.SetAddObj(obj: TBaseMathPersistence);
begin
     if Assigned(fAddObj) then
        fAddObj.Free;

     fAddObj := obj;
end;

end.
