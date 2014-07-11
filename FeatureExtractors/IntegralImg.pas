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

unit IntegralImg;

// ########################################################
// #### Integral Image creation
// ########################################################

interface

uses Matrix;

type
  TIntegralType = (itSum, itSumSqr, itSumSqrTilted);

// ########################################################
// #### Integral image creation - the algorithm for the tilted
// features are from: "An extended set of Haar-like features for Rapid Object Detection", Lienhart, Maydt
type
 TIntegralImage = class(TObject)
 private
   fSum : IMatrix;
   fSumSqr : IMatrix;
   fTilted : IMatrix;
   fNumColorPlanes : integer;
   fWidth, fHeight : integer;

   function InternalRecSum(sum : IMatrix; colPlane : integer; x, y, w, h : integer) : double;
   function InternalTiltRecSum(sum : IMatrix; colPlane : integer; x, y, w, h : integer) : double;

   procedure BuildSumImg(img : IMatrix; offset : integer);
   procedure BuildSumSqrImg(img : IMatrix; offset : integer);
   procedure BuildTiltedImg(img : IMatrix; offset : integer);

   //procedure BuildTiltedImgCV(img: IMatrix; offset: integer);
 public
   property NumColorPlanes : integer read fNumColorPlanes;
   property Width : integer read fWidth;
   property Height : integer read fHeight;

   function RecSum(x, y, w, h : integer) : double; overload;
   function RecSumSQR(x, y, w, h : integer) : double; overload;
   function TiltRecSum(x, y, w, h : integer) : double; overload;
   function RecSum(colPlane, x, y, w, h : integer) : double; overload;
   function RecSumSQR(colPlane, x, y, w, h : integer) : double; overload;
   function TiltRecSum(colPlane, x, y, w, h : integer) : double; overload;

   constructor Create(img : IMatrix; integralTypes : TIntegralType; numColPlanes : integer);
 end;

implementation

uses Math, Types, MatrixConst;

{ TIntegralImage }

procedure TIntegralImage.BuildSumImg(img: IMatrix; offset : integer);
var pData : PConstDoubleArr;
    pImg : PConstDoubleArr;
    pDataPrev : PConstDoubleArr;
    imgLineWidth : integer;
    lineWidth : integer;
    x, y : integer;
    lastVal : double;
begin
     if not Assigned(fSum) then
        fSum := TDoubleMatrix.Create(img.Width, img.Height);

     pImg := PConstDoubleArr(img.StartElement);
     imgLineWidth := img.LineWidth;
     inc(PByte(pImg), offset*imgLineWidth);

     pData := PConstDoubleArr(fSum.StartElement);
     lineWidth := fSum.LineWidth;
     inc(PByte(pData), lineWidth*offset);

     lastVal := 0;
     // first line is different
     for x := 0 to img.Width - 1 do
     begin
          pData^[x] := lastVal + pImg^[x];
          lastVal := pData^[x];
     end;

     pDataPrev := pData;
     inc(PByte(pData), lineWidth);
     inc(PByte(pImg), lineWidth);

     for y := 1 to img.Height - 1 do
     begin
          lastVal := 0;

          for x := 0 to img.Width - 1 do
          begin
               lastVal := lastVal + pImg^[x];
               pData^[x] := lastVal + pDataPrev^[x];
          end;

          pDataPrev := pData;
          inc(PByte(pData), lineWidth);
          inc(PByte(pImg), lineWidth);
     end;
end;

procedure TIntegralImage.BuildSumSqrImg(img: IMatrix; offset : integer);
var pImg : PConstDoubleArr;
    imgLineWidth : integer;
    pData : PConstDoubleArr;
    pDataPrev : PConstDoubleArr;
    lineWidth : integer;
    pDataSqr : PConstDoubleArr;
    pDataSqrPrev : PConstDoubleArr;
    lineWidthSqr : integer;
    x, y : integer;
    lastVal : double;
    lastValSQR : double;
begin
     if not Assigned(fSum) then
        fSum := TDoubleMatrix.Create(img.Width, img.Height);
     if not Assigned(fSumSqr) then
        fSumSqr := TDoubleMatrix.Create(img.Width, img.Height);

     pImg := PConstDoubleArr(img.StartElement);
     imgLineWidth := img.LineWidth;
     inc(PByte(pImg), imgLineWidth*offset);

     pData := PConstDoubleArr(fSum.StartElement);
     lineWidth := fSum.LineWidth;
     inc(PByte(pData), lineWidth*offset);

     pDataSqr := PConstDoubleArr(fSumSqr.StartElement);
     lineWidthSqr := fSumSqr.LineWidth;
     inc(PByte(pDataSqr), lineWidthSqr*offset);

     lastValSQR := 0;
     lastVal := 0;

     // first line is different
     for x := 0 to img.Width - 1 do
     begin
          lastVal := lastVal + pImg^[x];
          pData^[x] := lastVal;
          lastValSQR := lastValSQR + sqr(pImg^[x]);
          pDataSQR^[x] := lastValSQR;
     end;

     pDataPrev := pData;
     pDataSqrPrev := pDataSqr;
     inc(PByte(pData), lineWidth);
     inc(PByte(pDataSqr), lineWidthSqr);
     inc(PByte(pImg), lineWidth);

     for y := 1 to img.Height div fNumColorPlanes - 1 do
     begin
          lastValSQR := 0;
          lastVal := 0;

          for x := 0 to img.Width - 1 do
          begin
               lastVal := lastVal + pImg^[x];
               pData^[x] := lastVal + pDataPrev^[x];

               lastValSQR := lastValSQR + sqr(pImg^[x]);
               pDataSQR^[x] := lastValSQR + pDataSqrPrev^[x];
          end;

          pDataPrev := pData;
          pDataSqrPrev := pDataSqr;

          inc(PByte(pData), lineWidth);
          inc(PByte(pDataSqr), lineWidthSqr);
          inc(PByte(pImg), lineWidth);
     end;
end;

procedure TIntegralImage.BuildTiltedImg(img: IMatrix; offset: integer);
var pImg : PConstDoubleArr;
    imgLineWidth : integer;
    pData : PConstDoubleArr;
    lineWidth : integer;
    pdataPrev : PConstDoubleArr;
    x, y : integer;
begin
     assert((img.Width > 2) and (img.Height > 2), 'Error minimal image dimensions are 3x3');

     if not Assigned(fTilted) then
        fTilted := TDoubleMatrix.Create(img.Width, img.Height);

     pImg := PConstDoubleArr(img.StartElement);
     imgLineWidth := img.LineWidth;
     inc(PByte(pImg), imgLineWidth*offset);

     pData := PConstDoubleArr(fTilted.StartElement);
     lineWidth := fTilted.LineWidth;

     pDataPrev := pData;
     dec(PByte(pDataPrev), lineWidth);

     // first run RSAT(x,y) = RSAT(x-1,y-1) + RSAT(x-1,y) + I(x,y) - RSAT(x-2,y-1)

     // first line is different
     for x := 0 to img.Width - 1 do
         pData^[x] := pImg^[x];

     for y := 1 to img.Height - 1 do
     begin
          inc(PByte(pImg), imgLineWidth);
          inc(PByte(pData), lineWidth);
          inc(PByte(pdataPrev), lineWidth);

          // first two elements are differnt
          pData^[0] := pImg^[0];
          pData^[1] := pdataPrev^[0] + pData^[0] + pImg^[1];

          for x := 2 to img.Width - 1 do
              pData^[x] := pdataPrev^[x - 1] + pData^[x - 1] + pImg^[x] - pdataPrev^[x - 2];
     end;

     // second run RSAT(x,y) = RSAT(x,y) + RSAT(x-1,y+1) - RSAT(x-2,y) from bottom right to top left

     // last line is a bit different
     for x := img.Width - 1 downto 2 do
         pData^[x] := pData^[x] - pData^[x - 2];

     inc(PByte(pdataPrev), 2*lineWidth);

     for y := img.Height - 2 to 0 do
     begin
          dec(PByte(pData), lineWidth);
          dec(PByte(pdataPrev), lineWidth);

          for x := img.Width - 1 downto 2 do
              pData^[x] := pData^[x] + pdataPrev^[x - 1] - pData^[x - 2];

          // first two elements are different
          pData^[1] := pData^[1] + pDataPrev^[0];
     end;
end;

(*
procedure TIntegralImage.BuildTiltedImgCV(img: IMatrix; offset : integer);
var pImg : PConstDoubleArr;
    imgLineWidth : integer;
    pData : PConstDoubleArr;
    lineWidth : integer;
    pDataSqr : PConstDoubleArr;
    lineWidthSqr : integer;
    pDataTilt : PConstDoubleArr;
    lineWidthTilt : integer;
    x, y : integer;
    lastVal : double;
    lastValSQR : double;
    actVal : double;
    buf : Array of double;
    bufIdx : integer;
    t0, t1 : double;
    pTiltPrev : PConstDoubleArr;
begin
     if not Assigned(fSum) then
        fSum := TDoubleMatrix.Create(img.Width, img.Height);
     if not Assigned(fSumSqr) then
        fSumSqr := TDoubleMatrix.Create(img.Width, img.Height);
     if not Assigned(fTilted) then
        fTilted := TDoubleMatrix.Create(img.Width, img.Height);

     SetLength(buf, img.Width + 2);

     pImg := PConstDoubleArr(img.StartElement);
     imgLineWidth := img.LineWidth;
     inc(PByte(pImg), imgLineWidth*offset);

     pDataTilt := PConstDoubleArr(fTilted.StartElement);
     lineWidthTilt := fTilted.LineWidth;
     inc(PByte(pDataTilt), lineWidthTilt*offset);

     // first line
     for x := 0 to img.Width - 1 do
     begin
          pDataTilt^[x] := pImg^[x];
          buf[x] := pImg^[x];
     end;

     bufIdx := 0;
     pTiltPrev := pDataTilt;
     inc(PByte(pDataTilt), lineWidthTilt);

     for y := 1 to img.Height div fNumColorPlanes - 1 do
     begin
          //dec(bufIdx);
          t0 := pImg^[0];
          //pDataTilt[-cn] = tilted[-tiltedstep];
          pTiltPrev^[img.Width - 1] := pTiltPrev^[0];

          // tilted[0] = tilted[-tiltedstep] + t0 + buf[cn];
          pDataTilt^[0] := pTiltPrev^[0] + t0 + buf[1];

          for x := 1 to img.Width - 2 do
          begin
               t1 := buf[bufIdx + x];
               buf[x - 1] := t1 + t0;
               t0 := pImg^[x];
               t1 := t1 + buf[x + 1] + t0 + pTiltPrev^[x - 1];
               pDataTilt^[x] := t1;
          end;

          if( img.width > 1 ) then
          begin
               t1 := buf[img.Width - 1];
               buf[img.Width - 2] := t1 + t0;
               t0 := pImg^[img.Width - 1];
               pDataTilt^[img.Width - 1] := t0 + t1 + pTiltPrev^[img.Width - 2];
               buf[img.Width - 1] := t0;
          end;

          pTiltPrev := pDataTilt;
          inc(PByte(pDataTilt), lineWidthTilt);
          inc(PByte(pImg), lineWidth);
     end;
end;
*)

constructor TIntegralImage.Create(img: IMatrix; integralTypes : TIntegralType; numColPlanes : integer);
var offset : integer;
  i: Integer;
begin
     inherited Create;


     fNumColorPlanes := numColPlanes;
     fWidth := img.Width;
     fHeight := img.Height div fNumColorPlanes;

     for i := 0 to fNumColorPlanes - 1 do
     begin
          offset := (img.Height div fNumColorPlanes) * i;

          if integralTypes = itSum
          then
              BuildSumImg(img, offset)
          else if integralTypes = itSumSqr
          then
              BuildSumSqrImg(img, offset)
          else
          if itSumSqrTilted = integralTypes then
          begin
               BuildSumSqrImg(img, offset);
               BuildTiltedImg(img, offset);
          end;
     end;
end;

// RecSum(r) = SAT(x-1,y-1) + SAT(x+w-1,y+h-1) - SAT(x-1,y+h-1)-SAT(x+w-1,y-1)
function TIntegralImage.InternalRecSum(sum: IMatrix; colPlane, x, y, w,
  h: integer): double;
var right, bottom : integer;
    yOffset : integer;
begin
     if not Assigned(sum) then
     begin
          Result := 1;
          exit;
     end;

     right := Min(fWidth - 1, x + w - 1);
     bottom := Min(fHeight - 1, y + h - 1);

     yoffset := colPlane*fHeight;

     if x <= 0 then
     begin
          if y > 0
          then
              Result := sum[right, yoffset + bottom] - sum[right, yoffset + y - 1]
          else
              Result := sum[right, yoffset + bottom];
     end
     else
     begin
          if y > 0
          then
              Result := sum[x - 1, yoffset + y - 1] + Sum[right, yoffset + bottom] - sum[x - 1, yoffset + bottom] - sum[right, yoffset + y - 1]
          else
              Result := sum[right, yoffset + bottom] - Sum[x - 1, yoffset + bottom];
     end;
end;

// RecSum(r) = RSAT(x+w,x+w) + RSAT(x-h,y+h) - RSAT(x,y) - RSAT(x+w-h,y+w+h)
function TIntegralImage.InternalTiltRecSum(sum: IMatrix; colPlane, x, y, w,
  h: integer): double;
var yOffset : integer;
begin
     yOffset := colPlane*fHeight;

     Result := sum[Min(x + w, fWidth - 1), yOffset + Min(y + w, fHeight - 1)];
     if x - h >= 0 then
        Result := Result + sum[x - h, yOffset + Min(y + h, fHeight - 1)];
     if (x >= 0) and (y >= 0) then
        Result := Result - sum[x, yOffset + y];
     if x + w - h > 0 then
        Result := Result - sum[Min(x + w - h, fWidth - 1), yOffset + Min(y + w + h, fHeight - 1)];
end;

function TIntegralImage.RecSum(x, y, w, h: integer): double;
begin
     assert(Assigned(fSum), 'Error integral image sum not initialized');

     Result := InternalRecSum(fSum, 0, x, y, w, h);
end;

function TIntegralImage.RecSumSQR(x, y, w, h: integer): double;
begin
     assert(Assigned(fSumSQR), 'Error integral image sum not initialized');

     Result := InternalRecSum(fSumSQR, 0, x, y, w, h);
end;

function TIntegralImage.TiltRecSum(x, y, w, h: integer): double;
begin
     assert(Assigned(fTilted), 'Error tilted features are not calculated');

     Result := InternalTiltRecSum(fTilted, 0, x, y, w, h);
end;

function TIntegralImage.RecSum(colPlane, x, y, w, h: integer): double;
begin
     assert(Assigned(fSum), 'Error integral image sum not initialized');

     Result := InternalRecSum(fSum, colPlane, x, y, w, h);
end;

function TIntegralImage.RecSumSQR(colPlane, x, y, w, h: integer): double;
begin
     assert(Assigned(fSum), 'Error integral image sum not initialized');

     Result := InternalRecSum(fSumSqr, colPlane, x, y, w, h);
end;

function TIntegralImage.TiltRecSum(colPlane, x, y, w, h: integer): double;
begin
     assert(Assigned(fTilted), 'Error tilted features are not calculated');

     Result := InternalTiltRecSum(fTilted, colPlane, x, y, w, h);
end;

end.
