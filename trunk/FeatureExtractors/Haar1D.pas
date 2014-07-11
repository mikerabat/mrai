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

unit Haar1D;

// #############################################################
// #### 1D Haar like feature extrator
// #### (for boosting algorithms)
// #############################################################

interface

uses SysUtils, Classes, BaseClassifier;

const cChunkSize = $FFFF;

type
  T1DHaarType = (htPMBar, htPMPBar);
  T1DHaarPatchRec = record
    HaarType : T1DHaarType;
    StartIdxOffset : integer;
    EndIdxOffset : integer;
    Thresh : double;
  end;

// #############################################################
// #### Creation and handling of Haar like features for 1D signals
type
  T1DHaarRepresentation = class(TObject)
  private
    fIntegralData : Array of Array[0..cChunkSize - 1] of integer;
    fOverflows : Array of Array of integer;
    fOverflowIdx : Array of integer;
    fOverflowAdd : Array of Array of integer;
    fChunkIdx : integer;
    fNumChunks : integer;
    fFirstSamp : integer;
    procedure AddElement(const value : integer); inline;
    function CalcIntegralSum(const chunk1, data1, chunk2, data2 : integer) : int64; inline;
    procedure CheckArrayLenghts(NumChunks : integer);
    function GetNumSamples: integer;
  public
    property FirstSamp : integer read fFirstSamp;
    property NumSamples : integer read GetNumSamples;
  
    function HaarValueAt(FromIdx, ToIdx : integer; HaarType : T1DHaarType) : integer;

    procedure SetData(const Data : Array of integer);
    procedure AppendData(const Data : Array of integer; NumKeepBlocks : integer = -1);
    procedure Clear;
  end;

// ###############################################################
// #### Creation and handling of Haar like features for 1D floating point signals
// Note: this object here delivers floating point values and no overflow checks
// are performed!
// but for each append the elements are recalculated such that the likelyhood of an
// overflow in very large data runs is minimized
type 
  T1DHaarRepresentationF = class(TObject)
  private
    fIntegralData : Array of Array[0..cChunkSize - 1] of double;
    fNumChunks : integer;
    fChunkIdx : integer;
    fFirstSamp : integer;
    procedure CheckArrayLenghts(NumChunks: integer); inline;
    procedure AddElement(const value: double); inline;
    function GetNumSamples: integer;
  public
    property NumSamples : integer read GetNumSamples;
    property FirstSamp : integer read fFirstSamp;
    
    function HaarValueAt(FromIdx, ToIdx : integer; HaarType : T1DHaarType) : double;

    procedure SetData(const Data : Array of Double);
    procedure AppendData(const Data : Array of Double; NumKeepBlocks : integer = -1; SubtractFirstElem : boolean = False);
    procedure Clear;
  end;


// #################################################################
// #### Patch list creator with load and safe methods.
// Also creates a list of features from a given time point
type
  T1DHaarFeatureVec = class(TCustomFeatureList)
  private
    fFeatureVec : array of double;
  protected
    function GetFeature(index : integer) : double; override;
    procedure SetFeature(index : integer; value : double); override;
  public
    procedure SetFeatureVec(const Feature : Array of Double); override;
    constructor Create(const FeatureVec : Array of double); overload;
  end;

type
  T1DHaarTypedFeatureVec = class(T1DHaarFeatureVec)
  private
    fFeatureType : Array of T1DHaarType;
    function GetHaarType(index: integer): T1DHaarType;
  public
    property FeatureHaarType[index : integer] : T1DHaarType read GetHaarType;

    constructor Create(const FeatureVec : Array of double; const FeatureType : Array of T1DHaarType);
  end;

type
  T1DHaarPatchCreatorF = class(TObject)
  private
    fData : T1DHaarRepresentationF;
    fBaseWindowSize : integer;
    function GetFirstSample: integer;
    function GetNumSamples: integer;       // base patch size
  public
    // returns only features used from the one in the files. The scale factor is used just scale the patch size
    // -> results in time scale independent feature extraction (broadens the signal)
    // this base function creates a complete list of the two base haar like features.
    // The size is (n-1)*n/2 + (n - 2)*(n - 1)/2
    function FeatureVec(SampleIdx : integer; const Scale : double = 1) : T1DHaarFeatureVec; virtual;

    procedure SetData(const Data : Array of double);
    procedure AppendData(const Data : Array of double; NumKeepBlocks : integer = -1; SubtractFirstElem : boolean = False);

    property FirstSample : integer read GetFirstSample;
    property NumSamples : integer read GetNumSamples;
    
    constructor Create(BaseWindowSize : integer);
    destructor Destroy; override;
  end;

type
  T1DHaarLearnedPatchCreatorF = class(T1DHaarPatchCreatorF)
  private
    fPatches : array of T1DHaarPatchRec;
  public
    // returns only the features from the set list of patches 
    function FeatureVec(SampleIdx : integer; const Scale : double = 1) : T1DHaarFeatureVec; override;
  
    constructor Create(BaseWindowSize : integer; const Patches : array of T1DHaarPatchRec);
  end;

implementation

uses Math;

{$IFOPT Q+}
{$DEFINE REMOVEOVERFLOW}
{$ENDIF}

{$IFOPT R+}
{$DEFINE REMOVERANGE}
{$ENDIF}

{ T1DHaarLikeFeatures }

procedure T1DHaarRepresentation.AddElement(const value: integer);
begin
     if fChunkIdx <> 0
     then
         fIntegralData[fNumChunks][fChunkIdx] := fIntegralData[fNumChunks][fChunkIdx - 1] + Value
     else
         fIntegralData[fNumChunks][fChunkIdx] := fIntegralData[fNumChunks - 1][cChunkSize - 1] + Value;
end;

procedure T1DHaarRepresentation.AppendData(const Data: array of integer; NumKeepBlocks : integer);
var dataIdx : integer;
    i : integer;
begin
     Assert((fNumChunks <> 0) or (fChunkIdx <> 0), 'Error Call SetData before AppendData');

     // ########################################################
     // #### Remove old data
     if (NumKeepBlocks <> -1) and (NumKeepBlocks < fNumChunks - 1) then
     begin
          inc(fFirstSamp, cChunkSize*(fNumChunks - NumKeepBlocks));
          for i := 0 to NumKeepBlocks - 1 do
              fIntegralData[i] := fIntegralData[fNumChunks - NumKeepBlocks + i];

          SetLength(fIntegralData, NumKeepBlocks + 1);
          fNumChunks := NumKeepBlocks;
     end;

     // ########################################################
     // #### add new data
     dataIdx := 0;
     while dataIdx <= High(Data) do
     begin
          // ############################################################
          // #### Integral data representation for fast feature calculation:
          {$R-}{$Q-}

          // fill actual chunk
          CheckArrayLenghts(fChunkIdx + (Length(data) - dataIdx) div cChunkSize);

          while (fChunkIdx < cChunkSize) and (dataIdx <= High(data)) do
          begin
               AddElement(Data[dataIdx]);

               // check for an overflow
               if ((Data[dataIdx] < 0) and (fIntegralData[fNumChunks][fChunkIdx] > MaxInt + Data[dataIdx])) or
                  ((Data[dataIdx] > 0) and (fIntegralData[fNumChunks][fChunkIdx] < -MaxInt + Data[dataIdx]))
               then
               begin
                    // set overflow marker
                    if fOverflowIdx[fNumChunks] + 1 >= Length(fOverflows[fNumChunks]) then
                    begin
                         SetLength(fOverflows[fNumChunks], Length(fOverflows[fNumChunks]) + 10);
                         SetLength(fOverflowAdd[fNumChunks], Length(fOverflows[fNumChunks]));
                    end;
                    fOverflows[fNumChunks][fOverflowIdx[fNumChunks]] := fChunkIdx;

                    if ((Data[dataIdx] > 0) and (fIntegralData[fNumChunks][fChunkIdx] < -MaxInt + Data[dataIdx])) 
                    then
                        fOverflowAdd[fNumChunks][fOverflowIdx[fNumChunks]] := -MaxInt
                    else
                        fOverflowAdd[fNumChunks][fOverflowIdx[fNumChunks]] := MaxInt;

                    inc(fOverflowIdx[fNumChunks]);
               end;

               inc(dataIdx);
               inc(fChunkIdx);
          end;
          {$IFDEF REMOVEOVERFLOW}
          {$Q+}
          {$ENDIF}

          {$IFDEF REMOVERANGE}
          {$R+}
          {$ENDIF}
          if fChunkIdx = cChunkSize then
          begin
               inc(fNumChunks);
               fChunkIdx := 0;
          end;
     end;

     inc(fNumChunks);
end;

function T1DHaarRepresentation.CalcIntegralSum(const chunk1, data1, chunk2,
  data2: integer): int64;
var i : integer;
    j : integer;
begin
     // calculate the integral sum in the given area and take also the
     // integer overflows into account
     Result := fIntegralData[chunk2][data2] - fIntegralData[chunk1][data1];

     // check overflows within chunks
     for i := chunk1 to chunk2 do
     begin
          for j := 0 to fOverflowIdx[i] - 1 do
              if ((i > chunk1) and (i < chunk2)) or 
                 ((i = chunk1) and (fOverflows[i][j] >= data1)) or
                 ((i = chunk2) and (fOverflows[i][j] <= data2))
              then
                  Result := Result + fOverflowAdd[i][j];
     end;
end;

procedure T1DHaarRepresentation.CheckArrayLenghts(NumChunks : integer);
begin
     if Length(fIntegralData) <= numChunks then
     begin
          SetLength(fIntegralData, Length(fIntegralData) + 10);
          SetLength(fOverflows, Length(fIntegralData));
          SetLength(fOverflowAdd, Length(fIntegralData));
          SetLength(fOverflowIdx, Length(fIntegralData));
     end;
end;

procedure T1DHaarRepresentation.Clear;
begin
     fNumChunks := 0;
     fIntegralData := nil;
     fOverflows := nil;
end;

function T1DHaarRepresentation.GetNumSamples: integer;
begin
     Result := Max(0, (fNumChunks - 1)*cChunkSize + fChunkIdx);
end;

function T1DHaarRepresentation.HaarValueAt(FromIdx, ToIdx: integer;
  HaarType: T1DHaarType): integer;
var chkIdx1, chkIdx2 : integer;
    dataIdx1, dataIdx2 : integer;
    chkIdx3, dataIdx3 : integer;
    chkIdx4, dataIdx4 : integer;
    tmpIdx : integer;
    integralVal : Int64;
begin
     dec(FromIdx, fFirstSamp);
     dec(ToIdx, fFirstSamp);
     assert((FromIdx < ToIdx) and (ToIdx < fNumChunks*cChunkSize + fChunkIdx), 'Error index out of bounds');

     // calculate the index in the chunk arrays
     chkIdx1 := FromIdx div cChunkSize;
     chkIdx2 := ToIdx div cChunkSize;
     dataIdx1 := FromIdx mod cChunkSize;
     dataIdx2 := ToIdx mod cChunkSize;

     // calculate integral values:
     if HaarType = htPMBar then
     begin
          // this feature is calculated as the sum from "FromIdx" to (FromIdx + ToIdx)/2
          // minus the sum from (FromIdx + ToIdx)/2 to ToIdx
          tmpIdx := FromIdx + (FromIdx + ToIdx) div 2;
          chkIdx3 := tmpIdx div cChunkSize;
          dataIdx3 := tmpIdx mod cChunkSize;

          integralVal := CalcIntegralSum(chkIdx1, dataIdx1, chkIdx3, dataIdx3) -
                         CalcIntegralSum(chkIdx3, dataIdx3, chkIdx2, dataIdx2);
     end
     else
     begin
          // This time the indices are divided in three equaly broad areas. The result
          // is calculated as the sum of the elements in the outer areas minus the sum
          // of the elements in the inner area
          tmpIdx := FromIdx + (FromIdx + ToIdx) div 3;
          chkIdx3 := tmpIdx div cChunkSize;
          dataIdx3 := tmpIdx mod cChunkSize;
          tmpIdx := FromIdx + 2*(FromIdx + ToIdx) div 3;
          chkIdx4 := tmpIdx div cChunkSize;
          dataIdx4 := tmpIdx mod cChunkSize;
          integralVal := CalcIntegralSum(chkIdx1, dataIdx1, chkIdx3, dataIdx3) + 
                         CalcIntegralSum(chkIdx4, dataIdx4, chkIdx2, dataIdx2) -
                         CalcIntegralSum(chkIdx3, dataIdx3, chkIdx4, dataIdx4);
     end;

     Result := Integer(integralVal);
end;

procedure T1DHaarRepresentation.SetData(const Data: array of integer);
begin
     Clear;
     SetLength(fIntegralData, 10);
     fChunkIdx := 1;
     fFirstSamp := -1;            // note the zero value at the beginning is used for better consistency
     fIntegralData[0][0] := 0;
     AppendData(Data);
end;

{ T1DHaarLikeFeaturesF }

procedure T1DHaarRepresentationF.CheckArrayLenghts(NumChunks : integer);
begin
     if Length(fIntegralData) <= numChunks then
        SetLength(fIntegralData, Length(fIntegralData) + 10);
end;

procedure T1DHaarRepresentationF.AddElement(const value: double);
begin
     if fChunkIdx <> 0
     then
         fIntegralData[fNumChunks][fChunkIdx] := fIntegralData[fNumChunks][fChunkIdx - 1] + Value
     else
         fIntegralData[fNumChunks][fChunkIdx] := fIntegralData[fNumChunks - 1][cChunkSize - 1] + Value;
end;


procedure T1DHaarRepresentationF.AppendData(const Data: array of Double;
  NumKeepBlocks: integer; SubtractFirstElem : boolean);
var dataIdx : integer;
    i : integer;
    firstElem : double;
begin
     Assert((fNumChunks <> 0) or (fChunkIdx <> 0), 'Error Call SetData before AppendData');

     // ########################################################
     // #### Remove old data
     if (NumKeepBlocks <> -1) and (NumKeepBlocks < fNumChunks - 1) then
     begin
          inc(fFirstSamp, cChunkSize*(fNumChunks - NumKeepBlocks));
          for i := 0 to NumKeepBlocks - 1 do
              fIntegralData[i] := fIntegralData[fNumChunks - NumKeepBlocks + i];

          SetLength(fIntegralData, NumKeepBlocks + 1);
          fNumChunks := NumKeepBlocks;

          // subtract the first element from each following element -> reduce the 
          // risk of overflows or unpredictable precissions losses
          if SubtractFirstElem then
          begin
               firstElem := fIntegralData[0][0];
               // todo: this would be perfect for sse instructions!
               for i := 0 to fNumChunks - 2 do
                   for dataIdx := 0 to cChunkSize - 1 do
                       fIntegralData[i][dataIdx] := fIntegralData[i][dataIdx] - firstElem;

               for dataIdx := 0 to fChunkIdx - 1 do
                   fIntegralData[fNumChunks - 1][dataIdx] := fIntegralData[fNumChunks - 1][dataIdx] - firstElem;
          end;
     end;

     // ########################################################
     // #### add new data
     dataIdx := 0;
     while dataIdx <= High(Data) do
     begin
          // ############################################################
          // #### Integral data representation for fast feature calculation:
          // fill actual chunk
          CheckArrayLenghts(fChunkIdx + (Length(data) - dataIdx) div cChunkSize);

          while (fChunkIdx < cChunkSize) and (dataIdx <= High(data)) do
          begin
               AddElement(Data[dataIdx]);
               inc(dataIdx);
               inc(fChunkIdx);
          end;

          if fChunkIdx = cChunkSize then
          begin
               inc(fNumChunks);
               fChunkIdx := 0;
          end;
     end;

     inc(fNumChunks);
end;

procedure T1DHaarRepresentationF.Clear;
begin
     fIntegralData := nil;
     fNumChunks := 0;
end;

function T1DHaarRepresentationF.GetNumSamples: integer;
begin
     Result := Max(0, (fNumChunks - 1)*cChunkSize + fChunkIdx);
end;

function T1DHaarRepresentationF.HaarValueAt(FromIdx, ToIdx: integer;
  HaarType: T1DHaarType): double;
var chkIdx1, chkIdx2 : integer;
    dataIdx1, dataIdx2 : integer;
    chkIdx3, dataIdx3 : integer;
    chkIdx4, dataIdx4 : integer;
    tmpIdx : integer;
begin
     dec(FromIdx, fFirstSamp);
     dec(ToIdx, fFirstSamp);
     assert((FromIdx < ToIdx) and (ToIdx < fNumChunks*cChunkSize + fChunkIdx), 'Error index out of bounds');

     // calculate the index in the chunk arrays
     chkIdx1 := FromIdx div cChunkSize;
     chkIdx2 := ToIdx div cChunkSize;
     dataIdx1 := FromIdx mod cChunkSize;
     dataIdx2 := ToIdx mod cChunkSize;

     // calculate integral values:
     if HaarType = htPMBar then
     begin
          // this feature is calculated as the sum from "FromIdx" to (FromIdx + ToIdx)/2
          // minus the sum from (FromIdx + ToIdx)/2 to ToIdx
          tmpIdx := FromIdx + (FromIdx + ToIdx) div 2;
          chkIdx3 := tmpIdx div cChunkSize;
          dataIdx3 := tmpIdx mod cChunkSize;

          Result := fIntegralData[chkIdx3][dataIdx3] - fIntegralData[chkIdx1][dataIdx1] -
                    (fIntegralData[chkIdx2][dataIdx2] - fIntegralData[chkIdx3][dataIdx3]);
     end
     else
     begin
          // This time the indices are divided in three equaly broad areas. The result
          // is calculated as the sum of the elements in the outer areas minus the sum
          // of the elements in the inner area
          tmpIdx := FromIdx + (FromIdx + ToIdx) div 3;
          chkIdx3 := tmpIdx div cChunkSize;
          dataIdx3 := tmpIdx mod cChunkSize;
          tmpIdx := FromIdx + 2*(FromIdx + ToIdx) div 3;
          chkIdx4 := tmpIdx div cChunkSize;
          dataIdx4 := tmpIdx mod cChunkSize;

          Result := fIntegralData[chkIdx3][dataIdx3] - fIntegralData[chkIdx1][dataIdx1] +
                    fIntegralData[chkIdx2][dataIdx2] - fIntegralData[chkIdx4][dataIdx4] -
                    (fIntegralData[chkIdx4][dataIdx4] - fIntegralData[chkIdx3][dataIdx3]);
     end;
end;

procedure T1DHaarRepresentationF.SetData(const Data: array of Double);
begin
     Clear;
     SetLength(fIntegralData, 10);
     fIntegralData[0][0] := 0;
     fChunkIdx := 1;
     fFirstSamp := -1;
     AppendData(Data);
end;

{ T1DHaarPatchCreatorF }

procedure T1DHaarPatchCreatorF.AppendData(const Data: array of double; NumKeepBlocks : integer; SubtractFirstElem : boolean);
begin
     fData.AppendData(Data, NumKeepBlocks, SubtractFirstElem);     
end;

constructor T1DHaarPatchCreatorF.Create(BaseWindowSize : integer);
begin
     inherited Create;

     fBaseWindowSize := BaseWindowSize;
     fData := T1DHaarRepresentationF.Create; 
end;

destructor T1DHaarPatchCreatorF.Destroy;
begin
     fData.Free;

     inherited;
end;

function T1DHaarPatchCreatorF.FeatureVec(SampleIdx: integer;
  const Scale: double): T1DHaarFeatureVec;
var vec : array of double;
    n : integer;
    i, j : integer;
    numElem : integer;
    startIdx : integer;
    actFeature : integer;
    featureTypes : Array of T1DHaarType;
begin
     Result := nil;
     n := Round(fBaseWindowSize*scale) - 1;

     // check borders
     if (SampleIdx - n div 2 < fData.FirstSamp) or (SampleIdx + n div 2 >= fData.NumSamples) then
        exit;

     SetLength(vec, ((n - 1)*(n)) div 2 + ((n - 2)*(n - 1)) div 2);
     SetLength(featureTypes, Length(vec));
     numElem := ((n - 1)*n) div 2;
     startIdx := SampleIdx - n div 2;
     actFeature := 0;
     // ######################################################
     // #### Create first half of the feature vector
     for i := startIdx to startIdx + numElem - 1 do
     begin
          for j := i + 1 to startIdx + numElem do
          begin
               vec[actFeature] := fData.HaarValueAt(i, j, htPMBar);
               featureTypes[actFeature] := htPMBar;
               inc(actFeature);
          end;
     end;

     // ######################################################
     // #### Second half
     for i := startIdx to startIdx + numElem - 2 do
     begin
          for j := i + 2 to startIdx + numElem do
          begin
               vec[actFeature] := fData.HaarValueAt(i, j, htPMPBar);
               featureTypes[actFeature] := htPMPBar;
               inc(actFeature);
          end;
     end;

     // ######################################################
     // #### Build resulting object
     Result := T1DHaarTypedFeatureVec.Create(vec, featureTypes);
end;

function T1DHaarPatchCreatorF.GetFirstSample: integer;
begin
     Result := fData.FirstSamp;
end;

function T1DHaarPatchCreatorF.GetNumSamples: integer;
begin
     Result := fData.GetNumSamples;
end;

procedure T1DHaarPatchCreatorF.SetData(const Data: array of double);
begin
     fData.SetData(Data);
end;

{ T1DHaarFeatureVec }

constructor T1DHaarFeatureVec.Create(const FeatureVec: array of double);
begin
     inherited Create;

     assert(Length(FeatureVec) > 0, 'No empty array allowed');

     // todo: Eventually remove the moves and use a reference counted array type (faster)
     SetLength(fFeatureVec, Length(FeatureVec));
     Move(FeatureVec[0], fFeatureVec[0], sizeof(double)*Length(fFeatureVec));
end;

function T1DHaarFeatureVec.GetFeature(index: integer): double;
begin
     Result := fFeatureVec[index];
end;

procedure T1DHaarFeatureVec.SetFeature(index: integer; value: double);
begin
     fFeatureVec[index] := Value;
end;

procedure T1DHaarFeatureVec.SetFeatureVec(const Feature: array of Double);
begin
     SetLength(fFeatureVec, Length(Feature));
     Move(Feature[0], fFeatureVec[0], Length(Feature)*sizeof(double));
     fFeatureVecLen := Length(Feature);
end;

{ T1DHaarLearnedPatchCreatorF }

constructor T1DHaarLearnedPatchCreatorF.Create(BaseWindowSize: integer;
  const Patches: array of T1DHaarPatchRec);
begin
     inherited Create(BaseWindowSize);

     SetLength(fPatches, Length(Patches));
     Move(Patches[0], fPatches[0], Length(Patches)*sizeof(Patches[0]));
end;

function T1DHaarLearnedPatchCreatorF.FeatureVec(SampleIdx: integer;
  const Scale: double): T1DHaarFeatureVec;
var vec : array of double;
    n : integer;
    startSamp, EndSamp : integer;
    actFeature : integer;
begin
     Result := nil;
     n := Round(fBaseWindowSize*scale) - 1;

     // check borders
     if (SampleIdx - n div 2 < fData.FirstSamp) or (SampleIdx + n div 2 >= fData.NumSamples) then
        exit;

     SetLength(vec, Length(fPatches));
     // ######################################################
     // #### create the feature list from the learned patches
     for actFeature := 0 to Length(fPatches) do
     begin
          StartSamp := Round(SampleIdx + fPatches[actFeature].StartIdxOffset*scale);
          EndSamp := Round(SampleIdx + fPatches[actFeature].EndIdxOffset*scale);

          assert((StartSamp >= fData.FirstSamp) and (EndSamp < fData.NumSamples), 'Error Haar creation index out of bounds');

          vec[actFeature] := fData.HaarValueAt(StartSamp, EndSamp, fPatches[actFeature].HaarType);
     end;

     // ######################################################
     // #### Buld result
     Result := T1DHaarFeatureVec.Create;
     Result.SetFeatureVec(vec);
end;

{ T1DHaarLearnerFeatureVec }

constructor T1DHaarTypedFeatureVec.Create(const FeatureVec: array of double;
  const FeatureType: array of T1DHaarType);
begin
     assert(Length(FeatureVec) = Length(FeatureType), 'Error: Feature length does not match type length');
     inherited Create(FeatureVec);

     SetLength(fFeatureType, Length(fFeatureType));
     Move(FeatureType[0], fFeatureType[0], sizeof(T1DHaarType)*Length(fFeatureType));
end;

function T1DHaarTypedFeatureVec.GetHaarType(index: integer): T1DHaarType;
begin
     Result := fFeatureType[index];
end;

end.
