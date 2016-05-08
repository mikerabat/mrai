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

unit EnsembleClassifier;

// #############################################################
// #### Base ensemble algorithm stub
// #############################################################

interface

uses SysUtils, Classes, BaseClassifier, contnrs, BaseMathPersistence;

// #############################################################
// #### List of classifiers - used in all ensemble classifications
type
  TClassifierSet = class(TObjectList)
  private
    function GetClassifiers(index: integer): TCustomClassifier;
    procedure SetClassifiers(index: integer; const Value: TCustomClassifier);
  public
    procedure AddClassifier(classifier : TCustomClassifier);
    property Classifiers[index : integer] : TCustomClassifier read GetClassifiers write SetClassifiers; default;
  end;

// #############################################################
// #### Base ensamble classifier
type
  TEnsembelClassifier = class(TCustomClassifier)
  private
    fOwnsSet : boolean;
    fClassifiers : TClassifierSet;
    fIsInList : boolean;
  public
    property Classifiers : TClassifierSet read fClassifiers;
    
    // reader writer routines
    procedure DefineProps; override;
    function PropTypeOfName(const Name : string) : TPropType; override;

    function OnLoadObject(Obj : TBaseMathPersistence) : boolean; override;
    procedure OnLoadBeginList(const Name : String; count : integer); override;
    procedure OnLoadEndList; override;

    constructor Create(classifierSet : TClassifierSet; ownsSet : boolean);
    destructor Destroy; override;
  end;

implementation

{ TEnsembelClassifier }

const cEnsembleListBeginProp = 'ensemblelistbegin';
      cEnsembleListEndProp = 'ensemblelistend';

constructor TEnsembelClassifier.Create(classifierSet : TClassifierSet; ownsSet : boolean);
begin
     inherited Create;

     fOwnsSet := ownsSet;
     fClassifiers := classifierSet;
     assert(assigned(fClassifiers) and (fClassifiers.Count > 0), 'Ensemble methods without any classifier is not possible');
end;

procedure TEnsembelClassifier.DefineProps;
var i : integer;
begin
     BeginList(cEnsembleListBeginProp, fClassifiers.Count);
     for i := 0 to fClassifiers.Count - 1 do
         AddObject(fClassifiers[i]);
     EndList;
end;

function TEnsembelClassifier.PropTypeOfName(const Name: string): TPropType;
begin
     if CompareText(Name, cEnsembleListBeginProp) = 0
     then
         Result := ptObject
     else
         Result := inherited PropTypeOfName(Name);

end;


destructor TEnsembelClassifier.Destroy;
begin
     if fOwnsSet then
        FreeAndNil(fClassifiers);

     inherited;
end;

procedure TEnsembelClassifier.OnLoadBeginList(const Name: String;
  count: integer);
begin
     fIsInList := False;
     if CompareText(Name, cEnsembleListBeginProp) = 0 then
     begin
          fIsInList := True;
          assert(not Assigned(fClassifiers), 'Error list may not be loaded twice');
          fClassifiers := TClassifierSet.Create(True);
          fOwnsSet := True;
          fClassifiers.Capacity := Count;
     end
     else
         inherited;
end;

procedure TEnsembelClassifier.OnLoadEndList;
begin
     // do nothing here
     inherited;
end;

function TEnsembelClassifier.OnLoadObject(Obj: TBaseMathPersistence) : boolean;
begin
     if fIsInList then
     begin
          Result := True;
          assert(Assigned(fClassifiers), 'Error BeginList has not been called');
          fClassifiers.AddClassifier(Obj as TCustomClassifier);
     end
     else
         Result := inherited OnLoadObject(obj);
end;

{ TClassifierSet }

procedure TClassifierSet.AddClassifier(classifier: TCustomClassifier);
begin
     inherited Add(classifier);
end;

function TClassifierSet.GetClassifiers(index: integer): TCustomClassifier;
begin
     Result := Items[index] as TCustomClassifier;
end;

procedure TClassifierSet.SetClassifiers(index: integer;
  const Value: TCustomClassifier);
begin
     Items[index] := Value;
end;

end.
