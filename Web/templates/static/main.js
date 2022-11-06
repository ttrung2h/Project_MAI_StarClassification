// var resultPredict = "";
// var predictStar = "";
// const predictContainerImage = document.getElementById(
//   "image-predict-container"
// );
// const predictContainer = document.getElementById("form-predict_container");
// const form = document.getElementById("form-predict");
// console.log(form);
// var displayPredict = 0;
// resultPredict =
//   document.getElementsByClassName("result_predict")[0].textContent;
// predictStar = resultPredict.split("'");

// console.log(resultPredict);

// // form.addEventListener("submit", function (event) {
// //   // event.preventDefault();
// //   displayPredict = 1;
// //   console.log(123);

// //   setTimeout(() => {
// //     test();
// //   }, 5000);
// // });

// const closePredict = document.getElementById("btn-close");

// closePredict.addEventListener("click", function (e) {
//   predictContainerImage.style.display = "none";
//   predictContainerImage.style.visibility = "hidden";
//   predictContainer.style.display = "block";
//   predictContainer.style.visibility = "visible";

//   // window.location.reload();
// });

// const handleSubmit = (event) => {
//   predictContainerImage.style.display = "flex";
//   predictContainerImage.style.visibility = "visible";
//   predictContainer.style.display = "none";
//   predictContainer.style.visibility = "hidden";

//   if (predictStar[1] == "Main Sequence") {
//     document.getElementById("result_detail-img").src =
//       "../static/img/main_sequence.jpg";
//     document.getElementById("result_detail").innerHTML = "Main Sequence";
//     document.getElementById("link-attribute").href =
//       "https://en.wikipedia.org/wiki/Main_sequence";
//   } else if (predictStar[1] == "Red Dwarf") {
//     document.getElementById("result_detail-img").src =
//       "../static/img/Red_Dwarf.jpg";
//     document.getElementById("result_detail").innerHTML = "Red Dwarf";
//     document.getElementById("link-attribute").href =
//       "https://en.wikipedia.org/wiki/Red_dwarf";
//   } else if (predictStar[1] == "Brown Dwarf") {
//     document.getElementById("result_detail-img").src =
//       "../static/img/Brown_dwarf.jpg";
//     document.getElementById("result_detail").innerHTML = "Brown Dwarf";
//     document.getElementById("link-attribute").href =
//       "https://en.wikipedia.org/wiki/Brown_dwarf";
//   } else if (predictStar[1] == "White Dwarf") {
//     document.getElementById("result_detail-img").src =
//       "../static/img/White_dwarf.png";
//     document.getElementById("result_detail").innerHTML = "White Dwarf";
//     document.getElementById("link-attribute").href =
//       "https://en.wikipedia.org/wiki/White_dwarf";
//   } else if (predictStar[1] == "Super Giants") {
//     document.getElementById("result_detail-img").src =
//       "../static/img/super_giant.png";
//     document.getElementById("result_detail").innerHTML = "Super Giants";
//     document.getElementById("link-attribute").href =
//       "https://en.wikipedia.org/wiki/Supergiant";
//   } else if (predictStar[1] == "Hyper Giants") {
//     document.getElementById("result_detail-img").src =
//       "../static/img/hyper_giant.png";
//     document.getElementById("result_detail").innerHTML = "Hyper Giants";
//     document.getElementById("link-attribute").href =
//       "https://en.wikipedia.org/wiki/Hypergiant";
//   }
//   event.preventDefault();
// };
