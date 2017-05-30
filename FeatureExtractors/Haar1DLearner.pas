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

unit Haar1DLearner;

// ##########################################################
// #### Learner class which can be used to learn patterns with 1D Haar like
// #### features. (1D version of Viola Jones paper)
// #### This version can only distinguish between two classes
// ##########################################################

interface

uses SysUtils, Classes, BaseClassifier, contnrs, Haar1D;

// ##########################################################
// #### Signal stream to example list converter for 1D Haar features
type
  T1DHaarExampleCreator = class(TObject)
  private
    fExampleList : TObjectList;
    fData : T1DHaarPatchCreatorF;
  public
    // input of positive and negative examples
    // the indices belong to the complete data set (FirstSample offset)
    procedure AddData(const Data : Array of double; const PosExmplIdx : Array of integer;
                      NegExmplIdx : Array of integer);
    // this time the indices are given as relative offset to the given data.
    procedure AddDataRelIdx(const Data : Array of double; const PosExmplIdx : Array of integer;
                            NegExmplIdx : Array of integer);

    // resulting function to create the lerner object from the so far inputed data.
    function BuildLerningExmapleList(OnlyReference : boolean) : TCustomLearnerExampleList;

    constructor Create;
    destructor Destroy; override;
  end;


implementation

type
  T1DHaarExample = class(TCustomLearnerExample)
  public
    function CloneHaar(OnlyReference : boolean) : T1DHaarExample;
  end;

{ T1DHaarExampleCreator }

procedure T1DHaarExampleCreator.AddData(const Data: array of double;
  const PosExmplIdx: array of integer; NegExmplIdx: array of integer);
var featureVec : T1DHaarFeatureVec;
    i : integer;
    example : T1DHaarExample;
    numSampsToAdd : integer;
begin
     fData.AppendData(Data, 1, True);
     numSampsToAdd := fData.NumSamples - Length(data);

     // #########################################################
     // #### Create list of features from the given data
     for i := 0 to Length(PosExmplIdx) - 1 do
     begin
          // align indices such that it is compatible to the stored data
          featureVec := fData.FeatureVec(PosExmplIdx[i] + numSampsToAdd);

          if Assigned(featureVec) then
          begin
               example := T1DHaarExample.Create(featureVec, True);
               example.ClassVal := 1;
               fExampleList.Add(example);
          end;
     end;

     for i := 0 to Length(NegExmplIdx) - 1 do
     begin
          // align indices such that it is compatible to the stored data
          featureVec := fData.FeatureVec(NegExmplIdx[i] + numSampsToAdd);

          if Assigned(featureVec) then
          begin
               example := T1DHaarExample.Create(featureVec, True);
               example.ClassVal := -1;
               fExampleList.Add(example);
          end;
     end;
end;

procedure T1DHaarExampleCreator.AddDataRelIdx(const Data: array of double;
  const PosExmplIdx: array of integer; NegExmplIdx: array of integer);
var i : integer;
    PExmpl : Array of Integer;
    NExmpl : Array of integer;
begin
     SetLength(PExmpl, Length(PosExmplIdx));
     SetLength(NExmpl, Length(NegExmplIdx));

     // the data index is stored relative to the given block -> align to already read data
     for i := 0 to High(PosExmplIdx) do
         PExmpl[i] := PosExmplIdx[i] + fData.FirstSample + fData.NumSamples;
     for i := 0 to High(NegExmplIdx) do
         NExmpl[i] := NegExmplIdx[i] + fData.FirstSample + fData.NumSamples;

     AddData(Data, PExmpl, NExmpl);
end;

function T1DHaarExampleCreator.BuildLerningExmapleList(OnlyReference : boolean) : TCustomLearnerExampleList;
var i : integer;
begin
     Result := TCustomLearnerExampleList.Create;

     for i := 0 to fExampleList.Count - 1 do
         Result.Add(T1DHaarExample(fExampleList[i]).CloneHaar(OnlyReference));
end;

constructor T1DHaarExampleCreator.Create;
begin
     inherited Create;

     fExampleList := TObjectList.Create(True);
     fData := T1DHaarPatchCreatorF.Create(1);
end;

destructor T1DHaarExampleCreator.Destroy;
begin
     fExampleList.Free;
     fData.Free;

     inherited;
end;

{ T1DHaarExample }


function T1DHaarExample.CloneHaar(OnlyReference: boolean): T1DHaarExample;
var fVec : T1DHaarFeatureVec;
    features : Array of double;
    haarType : Array of T1DHaarType;
    i : Integer;
begin
     if not OnlyReference then
     begin
          assert(FeatureVec is T1DHaarTypedFeatureVec, 'Error wrong type for feature vector');

          SetLength(features, FeatureVec.FeatureVecLen);
          SetLength(haarType, FeatureVec.FeatureVecLen);
          for i := 0 to FeatureVec.FeatureVecLen - 1 do
          begin
               features[i] := FeatureVec.FeatureVec[i];
               haarType[i] := T1DHaarTypedFeatureVec(FeatureVec).FeatureHaarType[i];
          end;
          fVec := T1DHaarTypedFeatureVec.Create(features, haarType);

          Result := T1DHaarExample.Create(fVec, True);
     end
     else
         Result := T1DHaarExample.Create(FeatureVec, False);

     Result.ClassVal := ClassVal;
end;

end.
