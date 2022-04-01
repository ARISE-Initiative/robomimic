# Overview

<p align="center">
  <img width="24.0%" src="../images/task_lift.gif">
  <img width="24.0%" src="../images/task_can.gif">
  <img width="24.0%" src="../images/task_tool_hang.gif">
  <img width="24.0%" src="../images/task_square.gif">
  <img width="24.0%" src="../images/task_lift_real.gif">
  <img width="24.0%" src="../images/task_can_real.gif">
  <img width="24.0%" src="../images/task_tool_hang_real.gif">
  <img width="24.0%" src="../images/task_transport.gif">
 </p>

**robomimic** is a framework for robot learning from demonstration, with a focus on learning from human demonstrations.
It offers a broad set of demonstration datasets collected on robot manipulation domains and offline learning algorithms to learn from these datasets.
**robomimic** aims to make robot learning broadly *accessible* and *reproducible*, allowing researchers and practitioners to benchmark tasks and algorithms fairly and to develop the next generation of robot learning algorithms.

## Core Features

<style>
  @charset "UTF-8";
/*!
* Start Bootstrap - Agency v6.0.3 (https://startbootstrap.com/theme/agency)
* Copyright 2013-2020 Start Bootstrap
* Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-agency/blob/master/LICENSE)
*/
/*!
 * Bootstrap v4.5.3 (https://getbootstrap.com/)
 * Copyright 2011-2020 The Bootstrap Authors
 * Copyright 2011-2020 Twitter, Inc.
 * Licensed under MIT (https://github.com/twbs/bootstrap/blob/main/LICENSE)
 */
:root {
  --blue: #007bff;
  --indigo: #6610f2;
  --purple: #6f42c1;
  --pink: #e83e8c;
  --red: #dc3545;
  --orange: #fd7e14;
  --yellow: #fed136;
  --green: #28a745;
  --teal: #20c997;
  --cyan: #17a2b8;
  --white: #fff;
  --gray: #6c757d;
  --gray-dark: #343a40;
  --primary: #fed136;
  --secondary: #6c757d;
  --success: #28a745;
  --info: #17a2b8;
  --warning: #fed136;
  --danger: #dc3545;
  --light: #f8f9fa;
  --dark: #343a40;
  --breakpoint-xs: 0;
  --breakpoint-sm: 576px;
  --breakpoint-md: 768px;
  --breakpoint-lg: 992px;
  --breakpoint-xl: 1200px;
  --font-family-sans-serif: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji";
  --font-family-monospace: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
}

*,
*::before,
*::after {
  box-sizing: border-box;
}

.stat-box{
  background: gray;
  width: 100%;

}

html {
  font-family: sans-serif;
  line-height: 1.15;
  -webkit-text-size-adjust: 100%;
  -webkit-tap-highlight-color: rgba(0, 0, 0, 0);
}

article, aside, figcaption, figure, footer, header, hgroup, main, nav, section {
  display: block;
}



[tabindex="-1"]:focus:not(:focus-visible) {
  outline: 0 !important;
}

hr {
  box-sizing: content-box;
  height: 0;
  overflow: visible;
}

h1, h2, h3, h4, h5, h6 {
  margin-top: 0;
  margin-bottom: 0.5rem;
}

p {
  margin-top: 0;
  margin-bottom: 1rem;
}

abbr[title],
abbr[data-original-title] {
  text-decoration: underline;
  -webkit-text-decoration: underline dotted;
          text-decoration: underline dotted;
  cursor: help;
  border-bottom: 0;
  -webkit-text-decoration-skip-ink: none;
          text-decoration-skip-ink: none;
}

address {
  margin-bottom: 1rem;
  font-style: normal;
  line-height: inherit;
}

ol,
ul,
dl {
  margin-top: 0;
  margin-bottom: 1rem;
}

ol ol,
ul ul,
ol ul,
ul ol {
  margin-bottom: 0;
}

dt {
  font-weight: 700;
}

dd {
  margin-bottom: 0.5rem;
  margin-left: 0;
}

blockquote {
  margin: 0 0 1rem;
}

b,
strong {
  font-weight: bolder;
}

small {
  font-size: 80%;
}

sub,
sup {
  position: relative;
  font-size: 75%;
  line-height: 0;
  vertical-align: baseline;
}

sub {
  bottom: -0.25em;
}

sup {
  top: -0.5em;
}

a {
  color: #8C1515;
  text-decoration: none;
  background-color: transparent;
}
a:hover {
  color: #C21F1F;
  text-decoration: underline;
}

a:not([href]):not([class]) {
  color: inherit;
  text-decoration: none;
}
a:not([href]):not([class]):hover {
  color: inherit;
  text-decoration: none;
}

pre,
code,
kbd,
samp {
  font-family: SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  font-size: 1em;
}

pre {
  margin-top: 0;
  margin-bottom: 1rem;
  overflow: auto;
  -ms-overflow-style: scrollbar;
}

figure {
  margin: 0 0 1rem;
}

img {
  vertical-align: middle;
  border-style: none;
}

svg {
  overflow: hidden;
  vertical-align: middle;
}

table {
  border-collapse: collapse;
}

caption {
  padding-top: 0.75rem;
  padding-bottom: 0.75rem;
  color: #6c757d;
  text-align: left;
  caption-side: bottom;
}

th {
  text-align: inherit;
  text-align: -webkit-match-parent;
}

label {
  display: inline-block;
  margin-bottom: 0.5rem;
}

button {
  border-radius: 0;
}

button:focus {
  outline: 1px dotted;
  outline: 5px auto -webkit-focus-ring-color;
}

input,
button,
select,
optgroup,
textarea {
  margin: 0;
  font-family: inherit;
  font-size: inherit;
  line-height: inherit;
}

button,
input {
  overflow: visible;
}

button,
select {
  text-transform: none;
}

[role=button] {
  cursor: pointer;
}

select {
  word-wrap: normal;
}

button,
[type=button],
[type=reset],
[type=submit] {
  -webkit-appearance: button;
}

button:not(:disabled),
[type=button]:not(:disabled),
[type=reset]:not(:disabled),
[type=submit]:not(:disabled) {
  cursor: pointer;
}

button::-moz-focus-inner,
[type=button]::-moz-focus-inner,
[type=reset]::-moz-focus-inner,
[type=submit]::-moz-focus-inner {
  padding: 0;
  border-style: none;
}

input[type=radio],
input[type=checkbox] {
  box-sizing: border-box;
  padding: 0;
}

textarea {
  overflow: auto;
  resize: vertical;
}

fieldset {
  min-width: 0;
  padding: 0;
  margin: 0;
  border: 0;
}

legend {
  display: block;
  width: 100%;
  max-width: 100%;
  padding: 0;
  margin-bottom: 0.5rem;
  font-size: 1.5rem;
  line-height: inherit;
  color: inherit;
  white-space: normal;
}

progress {
  vertical-align: baseline;
}

[type=number]::-webkit-inner-spin-button,
[type=number]::-webkit-outer-spin-button {
  height: auto;
}

[type=search] {
  outline-offset: -2px;
  -webkit-appearance: none;
}

[type=search]::-webkit-search-decoration {
  -webkit-appearance: none;
}

::-webkit-file-upload-button {
  font: inherit;
  -webkit-appearance: button;
}

output {
  display: inline-block;
}

summary {
  display: list-item;
  cursor: pointer;
}

template {
  display: none;
}

[hidden] {
  display: none !important;
}

h1, h2, h3, h4, h5, h6,
.h1, .h2, .h3, .h4, .h5, .h6 {
  margin-bottom: 0.5rem;
  font-family: "Montserrat", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji";
  font-weight: 700;
  line-height: 1.2;
}

h1, .h1 {
  font-size: 2.5rem;
}

h2, .h2 {
  font-size: 2rem;
}

h3, .h3 {
  font-size: 1.75rem;
}

h4, .h4 {
  font-size: 1.5rem;
}

h5, .h5 {
  font-size: 1.25rem;
}

h6, .h6 {
  font-size: 1rem;
}

.lead {
  font-size: 1.25rem;
  font-weight: 300;
}

.display-1 {
  font-size: 6rem;
  font-weight: 300;
  line-height: 1.2;
}

.display-2 {
  font-size: 5.5rem;
  font-weight: 300;
  line-height: 1.2;
}

.display-3 {
  font-size: 4.5rem;
  font-weight: 300;
  line-height: 1.2;
}

.display-4 {
  font-size: 3.5rem;
  font-weight: 300;
  line-height: 1.2;
}

hr {
  margin-top: 1rem;
  margin-bottom: 1rem;
  border: 0;
  border-top: 1px solid rgba(0, 0, 0, 0.1);
}

small,
.small {
  font-size: 80%;
  font-weight: 400;
}

mark,
.mark {
  padding: 0.2em;
  background-color: #fcf8e3;
}

.list-unstyled {
  padding-left: 0;
  list-style: none;
}

.list-inline {
  padding-left: 0;
  list-style: none;
}

.list-inline-item {
  display: inline-block;
}
.list-inline-item:not(:last-child) {
  margin-right: 0.5rem;
}

.initialism {
  font-size: 90%;
  text-transform: uppercase;
}

.blockquote {
  margin-bottom: 1rem;
  font-size: 1.25rem;
}

.blockquote-footer {
  display: block;
  font-size: 80%;
  color: #6c757d;
}
.blockquote-footer::before {
  content: "— ";
}

.img-fluid {
  max-width: 100%;
  height: auto;
  margin: auto;
  display: block;
}

.img-thumbnail {
  padding: 0.25rem;
  background-color: #fff;
  border: 1px solid #dee2e6;
  border-radius: 0.25rem;
  max-width: 100%;
  height: auto;
}

.figure {
  display: inline-block;
}

.figure-img {
  margin-bottom: 0.5rem;
  line-height: 1;
}

.figure-caption {
  font-size: 90%;
  color: #6c757d;
}

code {
  font-size: 87.5%;
  color: #e83e8c;
  word-wrap: break-word;
}
a > code {
  color: inherit;
}

kbd {
  padding: 0.2rem 0.4rem;
  font-size: 87.5%;
  color: #fff;
  background-color: #212529;
  border-radius: 0.2rem;
}
kbd kbd {
  padding: 0;
  font-size: 100%;
  font-weight: 700;
}

pre {
  display: block;
  font-size: 87.5%;
  color: #212529;
}
pre code {
  font-size: inherit;
  color: inherit;
  word-break: normal;
}

.pre-scrollable {
  max-height: 340px;
  overflow-y: scroll;
}

.container,
.container-fluid,
.container-xl,
.container-lg,
.container-md,
.container-sm {
  width: 100%;
  padding-right: 0.75rem;
  padding-left: 0.75rem;
  margin-right: auto;
  margin-left: auto;
}

@media (min-width: 576px) {
  .container-sm, .container {
    max-width: 540px;
  }
}
@media (min-width: 768px) {
  .container-md, .container-sm, .container {
    max-width: 720px;
  }
}
@media (min-width: 992px) {
  .container-lg, .container-md, .container-sm, .container {
    max-width: 960px;
  }
}
@media (min-width: 1200px) {
  .container-xl, .container-lg, .container-md, .container-sm, .container {
    max-width: 1140px;
  }
}
.row {
  display: flex;
  flex-wrap: wrap;
  margin-right: -0.75rem;
  margin-left: -0.75rem;
}

.col-xl,
.col-xl-auto, .col-xl-12, .col-xl-11, .col-xl-10, .col-xl-9, .col-xl-8, .col-xl-7, .col-xl-6, .col-xl-5, .col-xl-4, .col-xl-3, .col-xl-2, .col-xl-1, .col-lg,
.col-lg-auto, .col-lg-12, .col-lg-11, .col-lg-10, .col-lg-9, .col-lg-8, .col-lg-7, .col-lg-6, .col-lg-5, .col-lg-4, .col-lg-3, .col-lg-2, .col-lg-1, .col-md,
.col-md-auto, .col-md-12, .col-md-11, .col-md-10, .col-md-9, .col-md-8, .col-md-7, .col-md-6, .col-md-5, .col-md-4, .col-md-3, .col-md-2, .col-md-1, .col-sm,
.col-sm-auto, .col-sm-12, .col-sm-11, .col-sm-10, .col-sm-9, .col-sm-8, .col-sm-7, .col-sm-6, .col-sm-5, .col-sm-4, .col-sm-3, .col-sm-2, .col-sm-1, .col,
.col-auto, .col-12, .col-11, .col-10, .col-9, .col-8, .col-7, .col-6, .col-5, .col-4, .col-3, .col-2, .col-1 {
  position: relative;
  width: 100%;
  padding-right: 0.75rem;
  padding-left: 0.75rem;
}

.col {
  flex-basis: 0;
  flex-grow: 1;
  max-width: 100%;
}

.row-cols-1 > * {
  flex: 0 0 100%;
  max-width: 100%;
}

.row-cols-2 > * {
  flex: 0 0 50%;
  max-width: 50%;
}

.row-cols-3 > * {
  flex: 0 0 33.3333333333%;
  max-width: 33.3333333333%;
}

.row-cols-4 > * {
  flex: 0 0 25%;
  max-width: 25%;
}

.row-cols-5 > * {
  flex: 0 0 20%;
  max-width: 20%;
}

.row-cols-6 > * {
  flex: 0 0 16.6666666667%;
  max-width: 16.6666666667%;
}

.col-auto {
  flex: 0 0 auto;
  width: auto;
  max-width: 100%;
}

.col-1 {
  flex: 0 0 8.3333333333%;
  max-width: 8.3333333333%;
}

.col-2 {
  flex: 0 0 16.6666666667%;
  max-width: 16.6666666667%;
}

.col-3 {
  flex: 0 0 25%;
  max-width: 25%;
}

.col-4 {
  flex: 0 0 33.3333333333%;
  max-width: 33.3333333333%;
}

.col-5 {
  flex: 0 0 41.6666666667%;
  max-width: 41.6666666667%;
}

.col-6 {
  flex: 0 0 50%;
  max-width: 50%;
}

.col-7 {
  flex: 0 0 58.3333333333%;
  max-width: 58.3333333333%;
}

.col-8 {
  flex: 0 0 66.6666666667%;
  max-width: 66.6666666667%;
}

.col-9 {
  flex: 0 0 75%;
  max-width: 75%;
}

.col-10 {
  flex: 0 0 83.3333333333%;
  max-width: 83.3333333333%;
}

.col-11 {
  flex: 0 0 91.6666666667%;
  max-width: 91.6666666667%;
}

.col-12 {
  flex: 0 0 100%;
  max-width: 100%;
}

@media (min-width: 576px) {
  .col-sm {
    flex-basis: 0;
    flex-grow: 1;
    max-width: 100%;
  }

  .row-cols-sm-1 > * {
    flex: 0 0 100%;
    max-width: 100%;
  }

  .row-cols-sm-2 > * {
    flex: 0 0 50%;
    max-width: 50%;
  }

  .row-cols-sm-3 > * {
    flex: 0 0 33.3333333333%;
    max-width: 33.3333333333%;
  }

  .row-cols-sm-4 > * {
    flex: 0 0 25%;
    max-width: 25%;
  }

  .row-cols-sm-5 > * {
    flex: 0 0 20%;
    max-width: 20%;
  }

  .row-cols-sm-6 > * {
    flex: 0 0 16.6666666667%;
    max-width: 16.6666666667%;
  }

  .col-sm-auto {
    flex: 0 0 auto;
    width: auto;
    max-width: 100%;
  }

  .col-sm-1 {
    flex: 0 0 8.3333333333%;
    max-width: 8.3333333333%;
  }

  .col-sm-2 {
    flex: 0 0 16.6666666667%;
    max-width: 16.6666666667%;
  }

  .col-sm-3 {
    flex: 0 0 25%;
    max-width: 25%;
  }

  .col-sm-4 {
    flex: 0 0 33.3333333333%;
    max-width: 33.3333333333%;
  }

  .col-sm-5 {
    flex: 0 0 41.6666666667%;
    max-width: 41.6666666667%;
  }

  .col-sm-6 {
    flex: 0 0 50%;
    max-width: 50%;
  }

  .col-sm-7 {
    flex: 0 0 58.3333333333%;
    max-width: 58.3333333333%;
  }

  .col-sm-8 {
    flex: 0 0 66.6666666667%;
    max-width: 66.6666666667%;
  }

  .col-sm-9 {
    flex: 0 0 75%;
    max-width: 75%;
  }

  .col-sm-10 {
    flex: 0 0 83.3333333333%;
    max-width: 83.3333333333%;
  }

  .col-sm-11 {
    flex: 0 0 91.6666666667%;
    max-width: 91.6666666667%;
  }

  .col-sm-12 {
    flex: 0 0 100%;
    max-width: 100%;
  }

  .order-sm-first {
    order: -1;
  }

  .order-sm-last {
    order: 13;
  }

  .order-sm-0 {
    order: 0;
  }

  .order-sm-1 {
    order: 1;
  }

  .order-sm-2 {
    order: 2;
  }

  .order-sm-3 {
    order: 3;
  }

  .order-sm-4 {
    order: 4;
  }

  .order-sm-5 {
    order: 5;
  }

  .order-sm-6 {
    order: 6;
  }

  .order-sm-7 {
    order: 7;
  }

  .order-sm-8 {
    order: 8;
  }

  .order-sm-9 {
    order: 9;
  }

  .order-sm-10 {
    order: 10;
  }

  .order-sm-11 {
    order: 11;
  }

  .order-sm-12 {
    order: 12;
  }

  .offset-sm-0 {
    margin-left: 0;
  }

  .offset-sm-1 {
    margin-left: 8.3333333333%;
  }

  .offset-sm-2 {
    margin-left: 16.6666666667%;
  }

  .offset-sm-3 {
    margin-left: 25%;
  }

  .offset-sm-4 {
    margin-left: 33.3333333333%;
  }

  .offset-sm-5 {
    margin-left: 41.6666666667%;
  }

  .offset-sm-6 {
    margin-left: 50%;
  }

  .offset-sm-7 {
    margin-left: 58.3333333333%;
  }

  .offset-sm-8 {
    margin-left: 66.6666666667%;
  }

  .offset-sm-9 {
    margin-left: 75%;
  }

  .offset-sm-10 {
    margin-left: 83.3333333333%;
  }

  .offset-sm-11 {
    margin-left: 91.6666666667%;
  }
}
@media (min-width: 768px) {
  .col-md {
    flex-basis: 0;
    flex-grow: 1;
    max-width: 100%;
  }

  .row-cols-md-1 > * {
    flex: 0 0 100%;
    max-width: 100%;
  }

  .row-cols-md-2 > * {
    flex: 0 0 50%;
    max-width: 50%;
  }

  .row-cols-md-3 > * {
    flex: 0 0 33.3333333333%;
    max-width: 33.3333333333%;
  }

  .row-cols-md-4 > * {
    flex: 0 0 25%;
    max-width: 25%;
  }

  .row-cols-md-5 > * {
    flex: 0 0 20%;
    max-width: 20%;
  }

  .row-cols-md-6 > * {
    flex: 0 0 16.6666666667%;
    max-width: 16.6666666667%;
  }

  .col-md-auto {
    flex: 0 0 auto;
    width: auto;
    max-width: 100%;
  }

  .col-md-1 {
    flex: 0 0 8.3333333333%;
    max-width: 8.3333333333%;
  }

  .col-md-2 {
    flex: 0 0 16.6666666667%;
    max-width: 16.6666666667%;
  }

  .col-md-3 {
    flex: 0 0 25%;
    max-width: 25%;
  }

  .col-md-4 {
    flex: 0 0 33.3333333333%;
    max-width: 33.3333333333%;
  }

  .col-md-5 {
    flex: 0 0 41.6666666667%;
    max-width: 41.6666666667%;
  }

  .col-md-6 {
    flex: 0 0 50%;
    max-width: 50%;
  }

  .col-md-7 {
    flex: 0 0 58.3333333333%;
    max-width: 58.3333333333%;
  }

  .col-md-8 {
    flex: 0 0 66.6666666667%;
    max-width: 66.6666666667%;
  }

  .col-md-9 {
    flex: 0 0 75%;
    max-width: 75%;
  }

  .col-md-10 {
    flex: 0 0 83.3333333333%;
    max-width: 83.3333333333%;
  }

  .col-md-11 {
    flex: 0 0 91.6666666667%;
    max-width: 91.6666666667%;
  }

  .col-md-12 {
    flex: 0 0 100%;
    max-width: 100%;
  }

  .order-md-first {
    order: -1;
  }

  .order-md-last {
    order: 13;
  }

  .order-md-0 {
    order: 0;
  }

  .order-md-1 {
    order: 1;
  }

  .order-md-2 {
    order: 2;
  }

  .order-md-3 {
    order: 3;
  }

  .order-md-4 {
    order: 4;
  }

  .order-md-5 {
    order: 5;
  }

  .order-md-6 {
    order: 6;
  }

  .order-md-7 {
    order: 7;
  }

  .order-md-8 {
    order: 8;
  }

  .order-md-9 {
    order: 9;
  }

  .order-md-10 {
    order: 10;
  }

  .order-md-11 {
    order: 11;
  }

  .order-md-12 {
    order: 12;
  }

  .offset-md-0 {
    margin-left: 0;
  }

  .offset-md-1 {
    margin-left: 8.3333333333%;
  }

  .offset-md-2 {
    margin-left: 16.6666666667%;
  }

  .offset-md-3 {
    margin-left: 25%;
  }

  .offset-md-4 {
    margin-left: 33.3333333333%;
  }

  .offset-md-5 {
    margin-left: 41.6666666667%;
  }

  .offset-md-6 {
    margin-left: 50%;
  }

  .offset-md-7 {
    margin-left: 58.3333333333%;
  }

  .offset-md-8 {
    margin-left: 66.6666666667%;
  }

  .offset-md-9 {
    margin-left: 75%;
  }

  .offset-md-10 {
    margin-left: 83.3333333333%;
  }

  .offset-md-11 {
    margin-left: 91.6666666667%;
  }
}
@media (min-width: 992px) {
  .col-lg {
    flex-basis: 0;
    flex-grow: 1;
    max-width: 100%;
  }

  .row-cols-lg-1 > * {
    flex: 0 0 100%;
    max-width: 100%;
  }

  .row-cols-lg-2 > * {
    flex: 0 0 50%;
    max-width: 50%;
  }

  .row-cols-lg-3 > * {
    flex: 0 0 33.3333333333%;
    max-width: 33.3333333333%;
  }

  .row-cols-lg-4 > * {
    flex: 0 0 25%;
    max-width: 25%;
  }

  .row-cols-lg-5 > * {
    flex: 0 0 20%;
    max-width: 20%;
  }

  .row-cols-lg-6 > * {
    flex: 0 0 16.6666666667%;
    max-width: 16.6666666667%;
  }

  .col-lg-auto {
    flex: 0 0 auto;
    width: auto;
    max-width: 100%;
  }

  .col-lg-1 {
    flex: 0 0 8.3333333333%;
    max-width: 8.3333333333%;
  }

  .col-lg-2 {
    flex: 0 0 16.6666666667%;
    max-width: 16.6666666667%;
  }

  .col-lg-3 {
    flex: 0 0 25%;
    max-width: 25%;
  }

  .col-lg-4 {
    flex: 0 0 33.3333333333%;
    max-width: 33.3333333333%;
  }

  .col-lg-5 {
    flex: 0 0 41.6666666667%;
    max-width: 41.6666666667%;
  }

  .col-lg-6 {
    flex: 0 0 50%;
    max-width: 50%;
  }

  .col-lg-7 {
    flex: 0 0 58.3333333333%;
    max-width: 58.3333333333%;
  }

  .col-lg-8 {
    flex: 0 0 66.6666666667%;
    max-width: 66.6666666667%;
  }

  .col-lg-9 {
    flex: 0 0 75%;
    max-width: 75%;
  }

  .col-lg-10 {
    flex: 0 0 83.3333333333%;
    max-width: 83.3333333333%;
  }

  .col-lg-11 {
    flex: 0 0 91.6666666667%;
    max-width: 91.6666666667%;
  }

  .col-lg-12 {
    flex: 0 0 100%;
    max-width: 100%;
  }

  .order-lg-first {
    order: -1;
  }

  .order-lg-last {
    order: 13;
  }

  .order-lg-0 {
    order: 0;
  }

  .order-lg-1 {
    order: 1;
  }

  .order-lg-2 {
    order: 2;
  }

  .order-lg-3 {
    order: 3;
  }

  .order-lg-4 {
    order: 4;
  }

  .order-lg-5 {
    order: 5;
  }

  .order-lg-6 {
    order: 6;
  }

  .order-lg-7 {
    order: 7;
  }

  .order-lg-8 {
    order: 8;
  }

  .order-lg-9 {
    order: 9;
  }

  .order-lg-10 {
    order: 10;
  }

  .order-lg-11 {
    order: 11;
  }

  .order-lg-12 {
    order: 12;
  }

  .offset-lg-0 {
    margin-left: 0;
  }

  .offset-lg-1 {
    margin-left: 8.3333333333%;
  }

  .offset-lg-2 {
    margin-left: 16.6666666667%;
  }

  .offset-lg-3 {
    margin-left: 25%;
  }

  .offset-lg-4 {
    margin-left: 33.3333333333%;
  }

  .offset-lg-5 {
    margin-left: 41.6666666667%;
  }

  .offset-lg-6 {
    margin-left: 50%;
  }

  .offset-lg-7 {
    margin-left: 58.3333333333%;
  }

  .offset-lg-8 {
    margin-left: 66.6666666667%;
  }

  .offset-lg-9 {
    margin-left: 75%;
  }

  .offset-lg-10 {
    margin-left: 83.3333333333%;
  }

  .offset-lg-11 {
    margin-left: 91.6666666667%;
  }
}
@media (min-width: 1200px) {
  .col-xl {
    flex-basis: 0;
    flex-grow: 1;
    max-width: 100%;
  }

  .row-cols-xl-1 > * {
    flex: 0 0 100%;
    max-width: 100%;
  }

  .row-cols-xl-2 > * {
    flex: 0 0 50%;
    max-width: 50%;
  }

  .row-cols-xl-3 > * {
    flex: 0 0 33.3333333333%;
    max-width: 33.3333333333%;
  }

  .row-cols-xl-4 > * {
    flex: 0 0 25%;
    max-width: 25%;
  }

  .row-cols-xl-5 > * {
    flex: 0 0 20%;
    max-width: 20%;
  }

  .row-cols-xl-6 > * {
    flex: 0 0 16.6666666667%;
    max-width: 16.6666666667%;
  }

  .col-xl-auto {
    flex: 0 0 auto;
    width: auto;
    max-width: 100%;
  }

  .col-xl-1 {
    flex: 0 0 8.3333333333%;
    max-width: 8.3333333333%;
  }

  .col-xl-2 {
    flex: 0 0 16.6666666667%;
    max-width: 16.6666666667%;
  }

  .col-xl-3 {
    flex: 0 0 25%;
    max-width: 25%;
  }

  .col-xl-4 {
    flex: 0 0 33.3333333333%;
    max-width: 33.3333333333%;
  }

  .col-xl-5 {
    flex: 0 0 41.6666666667%;
    max-width: 41.6666666667%;
  }

  .col-xl-6 {
    flex: 0 0 50%;
    max-width: 50%;
  }

  .col-xl-7 {
    flex: 0 0 58.3333333333%;
    max-width: 58.3333333333%;
  }

  .col-xl-8 {
    flex: 0 0 66.6666666667%;
    max-width: 66.6666666667%;
  }

  .col-xl-9 {
    flex: 0 0 75%;
    max-width: 75%;
  }

  .col-xl-10 {
    flex: 0 0 83.3333333333%;
    max-width: 83.3333333333%;
  }

  .col-xl-11 {
    flex: 0 0 91.6666666667%;
    max-width: 91.6666666667%;
  }

  .col-xl-12 {
    flex: 0 0 100%;
    max-width: 100%;
  }

  .order-xl-first {
    order: -1;
  }

  .order-xl-last {
    order: 13;
  }

  .order-xl-0 {
    order: 0;
  }

  .order-xl-1 {
    order: 1;
  }

  .order-xl-2 {
    order: 2;
  }

  .order-xl-3 {
    order: 3;
  }

  .order-xl-4 {
    order: 4;
  }

  .order-xl-5 {
    order: 5;
  }

  .order-xl-6 {
    order: 6;
  }

  .order-xl-7 {
    order: 7;
  }

  .order-xl-8 {
    order: 8;
  }

  .order-xl-9 {
    order: 9;
  }

  .order-xl-10 {
    order: 10;
  }

  .order-xl-11 {
    order: 11;
  }

  .order-xl-12 {
    order: 12;
  }

  .offset-xl-0 {
    margin-left: 0;
  }

  .offset-xl-1 {
    margin-left: 8.3333333333%;
  }

  .offset-xl-2 {
    margin-left: 16.6666666667%;
  }

  .offset-xl-3 {
    margin-left: 25%;
  }

  .offset-xl-4 {
    margin-left: 33.3333333333%;
  }

  .offset-xl-5 {
    margin-left: 41.6666666667%;
  }

  .offset-xl-6 {
    margin-left: 50%;
  }

  .offset-xl-7 {
    margin-left: 58.3333333333%;
  }

  .offset-xl-8 {
    margin-left: 66.6666666667%;
  }

  .offset-xl-9 {
    margin-left: 75%;
  }

  .offset-xl-10 {
    margin-left: 83.3333333333%;
  }

  .offset-xl-11 {
    margin-left: 91.6666666667%;
  }
}
.table {
  width: 100%;
  margin-bottom: 1rem;
  color: #212529;
}
.table th,
.table td {
  padding: 0.75rem;
  vertical-align: top;
  border-top: 1px solid #dee2e6;
}
.table thead th {
  vertical-align: bottom;
  border-bottom: 2px solid #dee2e6;
}
.table tbody + tbody {
  border-top: 2px solid #dee2e6;
}

.table-sm th,
.table-sm td {
  padding: 0.3rem;
}

.table-bordered {
  border: 1px solid #dee2e6;
}
.table-bordered th,
.table-bordered td {
  border: 1px solid #dee2e6;
}
.table-bordered thead th,
.table-bordered thead td {
  border-bottom-width: 2px;
}

.table-borderless th,
.table-borderless td,
.table-borderless thead th,
.table-borderless tbody + tbody {
  border: 0;
}

.table-striped tbody tr:nth-of-type(odd) {
  background-color: rgba(0, 0, 0, 0.05);
}

.table-hover tbody tr:hover {
  color: #212529;
  background-color: rgba(0, 0, 0, 0.075);
}

.table-primary,
.table-primary > th,
.table-primary > td {
  background-color: #fff2c7;
}
.table-primary th,
.table-primary td,
.table-primary thead th,
.table-primary tbody + tbody {
  border-color: #fee796;
}

.table-hover .table-primary:hover {
  background-color: #ffecae;
}
.table-hover .table-primary:hover > td,
.table-hover .table-primary:hover > th {
  background-color: #ffecae;
}

.table-secondary,
.table-secondary > th,
.table-secondary > td {
  background-color: #d6d8db;
}
.table-secondary th,
.table-secondary td,
.table-secondary thead th,
.table-secondary tbody + tbody {
  border-color: #b3b7bb;
}

.table-hover .table-secondary:hover {
  background-color: #c8cbcf;
}
.table-hover .table-secondary:hover > td,
.table-hover .table-secondary:hover > th {
  background-color: #c8cbcf;
}

.table-success,
.table-success > th,
.table-success > td {
  background-color: #c3e6cb;
}
.table-success th,
.table-success td,
.table-success thead th,
.table-success tbody + tbody {
  border-color: #8fd19e;
}

.table-hover .table-success:hover {
  background-color: #b1dfbb;
}
.table-hover .table-success:hover > td,
.table-hover .table-success:hover > th {
  background-color: #b1dfbb;
}

.table-info,
.table-info > th,
.table-info > td {
  background-color: #bee5eb;
}
.table-info th,
.table-info td,
.table-info thead th,
.table-info tbody + tbody {
  border-color: #86cfda;
}

.table-hover .table-info:hover {
  background-color: #abdde5;
}
.table-hover .table-info:hover > td,
.table-hover .table-info:hover > th {
  background-color: #abdde5;
}

.table-warning,
.table-warning > th,
.table-warning > td {
  background-color: #fff2c7;
}
.table-warning th,
.table-warning td,
.table-warning thead th,
.table-warning tbody + tbody {
  border-color: #fee796;
}

.table-hover .table-warning:hover {
  background-color: #ffecae;
}
.table-hover .table-warning:hover > td,
.table-hover .table-warning:hover > th {
  background-color: #ffecae;
}

.table-danger,
.table-danger > th,
.table-danger > td {
  background-color: #f5c6cb;
}
.table-danger th,
.table-danger td,
.table-danger thead th,
.table-danger tbody + tbody {
  border-color: #ed969e;
}

.table-hover .table-danger:hover {
  background-color: #f1b0b7;
}
.table-hover .table-danger:hover > td,
.table-hover .table-danger:hover > th {
  background-color: #f1b0b7;
}

.table-light,
.table-light > th,
.table-light > td {
  background-color: #fdfdfe;
}
.table-light th,
.table-light td,
.table-light thead th,
.table-light tbody + tbody {
  border-color: #fbfcfc;
}

.table-hover .table-light:hover {
  background-color: #ececf6;
}
.table-hover .table-light:hover > td,
.table-hover .table-light:hover > th {
  background-color: #ececf6;
}

.table-dark,
.table-dark > th,
.table-dark > td {
  background-color: #c6c8ca;
}
.table-dark th,
.table-dark td,
.table-dark thead th,
.table-dark tbody + tbody {
  border-color: #95999c;
}

.table-hover .table-dark:hover {
  background-color: #b9bbbe;
}
.table-hover .table-dark:hover > td,
.table-hover .table-dark:hover > th {
  background-color: #b9bbbe;
}

.table-active,
.table-active > th,
.table-active > td {
  background-color: rgba(0, 0, 0, 0.075);
}

.table-hover .table-active:hover {
  background-color: rgba(0, 0, 0, 0.075);
}
.table-hover .table-active:hover > td,
.table-hover .table-active:hover > th {
  background-color: rgba(0, 0, 0, 0.075);
}

.table .thead-dark th {
  color: #fff;
  background-color: #343a40;
  border-color: #454d55;
}
.table .thead-light th {
  color: #495057;
  background-color: #e9ecef;
  border-color: #dee2e6;
}

.table-dark {
  color: #fff;
  background-color: #343a40;
}
.table-dark th,
.table-dark td,
.table-dark thead th {
  border-color: #454d55;
}
.table-dark.table-bordered {
  border: 0;
}
.table-dark.table-striped tbody tr:nth-of-type(odd) {
  background-color: rgba(255, 255, 255, 0.05);
}
.table-dark.table-hover tbody tr:hover {
  color: #fff;
  background-color: rgba(255, 255, 255, 0.075);
}

@media (max-width: 575.98px) {
  .table-responsive-sm {
    display: block;
    width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }
  .table-responsive-sm > .table-bordered {
    border: 0;
  }
}
@media (max-width: 767.98px) {
  .table-responsive-md {
    display: block;
    width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }
  .table-responsive-md > .table-bordered {
    border: 0;
  }
}
@media (max-width: 991.98px) {
  .table-responsive-lg {
    display: block;
    width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }
  .table-responsive-lg > .table-bordered {
    border: 0;
  }
}
@media (max-width: 1199.98px) {
  .table-responsive-xl {
    display: block;
    width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }
  .table-responsive-xl > .table-bordered {
    border: 0;
  }
}
.table-responsive {
  display: block;
  width: 100%;
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
}
.table-responsive > .table-bordered {
  border: 0;
}

.feature-box {
    padding: 30px 20px;
    margin-bottom: 50px;
    text-align: center;
    border: 1px solid #e6e6e6;
    height: 400px;
    position: relative;
    background: #faf0f0;
}

.paper-box {
    padding: 60px 30px 20px 30px;
    margin-bottom: 20px;
    text-align: center;
    border: 1px solid #e6e6e6;
    min-height: 400px;
    overflow: auto;
    position: relative;
    display: block;
    background: #faf0f0;
}


.form-control {
  display: block;
  width: 100%;
  height: calc(1.5em + 0.75rem + 2px);
  padding: 0.375rem 0.75rem;
  font-size: 1rem;
  font-weight: 400;
  line-height: 1.5;
  color: #495057;
  background-color: #fff;
  background-clip: padding-box;
  border: 1px solid #ced4da;
  border-radius: 0.25rem;
  transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
}
@media (prefers-reduced-motion: reduce) {
  .form-control {
    transition: none;
  }
}
.form-control::-ms-expand {
  background-color: transparent;
  border: 0;
}
.form-control:-moz-focusring {
  color: transparent;
  text-shadow: 0 0 0 #495057;
}
.form-control:focus {
  color: #495057;
  background-color: #fff;
  border-color: #ffeeb5;
  outline: 0;
  box-shadow: 0 0 0 0.2rem rgba(254, 209, 54, 0.25);
}
.form-control::-moz-placeholder {
  color: #6c757d;
  opacity: 1;
}
.form-control:-ms-input-placeholder {
  color: #6c757d;
  opacity: 1;
}
.form-control::placeholder {
  color: #6c757d;
  opacity: 1;
}
.form-control:disabled, .form-control[readonly] {
  background-color: #e9ecef;
  opacity: 1;
}

input[type=date].form-control,
input[type=time].form-control,
input[type=datetime-local].form-control,
input[type=month].form-control {
  -webkit-appearance: none;
     -moz-appearance: none;
          appearance: none;
}

select.form-control:focus::-ms-value {
  color: #495057;
  background-color: #fff;
}

.form-control-file,
.form-control-range {
  display: block;
  width: 100%;
}

.col-form-label {
  padding-top: calc(0.375rem + 1px);
  padding-bottom: calc(0.375rem + 1px);
  margin-bottom: 0;
  font-size: inherit;
  line-height: 1.5;
}

.col-form-label-lg {
  padding-top: calc(0.5rem + 1px);
  padding-bottom: calc(0.5rem + 1px);
  font-size: 1.25rem;
  line-height: 1.5;
}

.col-form-label-sm {
  padding-top: calc(0.25rem + 1px);
  padding-bottom: calc(0.25rem + 1px);
  font-size: 0.875rem;
  line-height: 1.5;
}

.form-control-plaintext {
  display: block;
  width: 100%;
  padding: 0.375rem 0;
  margin-bottom: 0;
  font-size: 1rem;
  line-height: 1.5;
  color: #212529;
  background-color: transparent;
  border: solid transparent;
  border-width: 1px 0;
}
.form-control-plaintext.form-control-sm, .form-control-plaintext.form-control-lg {
  padding-right: 0;
  padding-left: 0;
}

.form-control-sm {
  height: calc(1.5em + 0.5rem + 2px);
  padding: 0.25rem 0.5rem;
  font-size: 0.875rem;
  line-height: 1.5;
  border-radius: 0.2rem;
}

.form-control-lg {
  height: calc(1.5em + 1rem + 2px);
  padding: 0.5rem 1rem;
  font-size: 1.25rem;
  line-height: 1.5;
  border-radius: 0.3rem;
}

select.form-control[size], select.form-control[multiple] {
  height: auto;
}

textarea.form-control {
  height: auto;
}

.form-group {
  margin-bottom: 1rem;
}

.form-text {
  display: block;
  margin-top: 0.25rem;
}

.form-row {
  display: flex;
  flex-wrap: wrap;
  margin-right: -5px;
  margin-left: -5px;
}
.form-row > .col,
.form-row > [class*=col-] {
  padding-right: 5px;
  padding-left: 5px;
}

.form-check {
  position: relative;
  display: block;
  padding-left: 1.25rem;
}

.form-check-input {
  position: absolute;
  margin-top: 0.3rem;
  margin-left: -1.25rem;
}
.form-check-input[disabled] ~ .form-check-label, .form-check-input:disabled ~ .form-check-label {
  color: #6c757d;
}

.form-check-label {
  margin-bottom: 0;
}

.form-check-inline {
  display: inline-flex;
  align-items: center;
  padding-left: 0;
  margin-right: 0.75rem;
}
.form-check-inline .form-check-input {
  position: static;
  margin-top: 0;
  margin-right: 0.3125rem;
  margin-left: 0;
}

.valid-feedback {
  display: none;
  width: 100%;
  margin-top: 0.25rem;
  font-size: 80%;
  color: #28a745;
}

.valid-tooltip {
  position: absolute;
  top: 100%;
  left: 0;
  z-index: 5;
  display: none;
  max-width: 100%;
  padding: 0.25rem 0.5rem;
  margin-top: 0.1rem;
  font-size: 0.875rem;
  line-height: 1.5;
  color: #fff;
  background-color: rgba(40, 167, 69, 0.9);
  border-radius: 0.25rem;
}

.was-validated :valid ~ .valid-feedback,
.was-validated :valid ~ .valid-tooltip,
.is-valid ~ .valid-feedback,
.is-valid ~ .valid-tooltip {
  display: block;
}

.was-validated .form-control:valid, .form-control.is-valid {
  border-color: #28a745;
  padding-right: calc(1.5em + 0.75rem);
  background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='8' height='8' viewBox='0 0 8 8'%3e%3cpath fill='%2328a745' d='M2.3 6.73L.6 4.53c-.4-1.04.46-1.4 1.1-.8l1.1 1.4 3.4-3.8c.6-.63 1.6-.27 1.2.7l-4 4.6c-.43.5-.8.4-1.1.1z'/%3e%3c/svg%3e");
  background-repeat: no-repeat;
  background-position: right calc(0.375em + 0.1875rem) center;
  background-size: calc(0.75em + 0.375rem) calc(0.75em + 0.375rem);
}
.was-validated .form-control:valid:focus, .form-control.is-valid:focus {
  border-color: #28a745;
  box-shadow: 0 0 0 0.2rem rgba(40, 167, 69, 0.25);
}

.was-validated textarea.form-control:valid, textarea.form-control.is-valid {
  padding-right: calc(1.5em + 0.75rem);
  background-position: top calc(0.375em + 0.1875rem) right calc(0.375em + 0.1875rem);
}

.was-validated .custom-select:valid, .custom-select.is-valid {
  border-color: #28a745;
  padding-right: calc(0.75em + 2.3125rem);
  background: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='4' height='5' viewBox='0 0 4 5'%3e%3cpath fill='%23343a40' d='M2 0L0 2h4zm0 5L0 3h4z'/%3e%3c/svg%3e") no-repeat right 0.75rem center/8px 10px, url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='8' height='8' viewBox='0 0 8 8'%3e%3cpath fill='%2328a745' d='M2.3 6.73L.6 4.53c-.4-1.04.46-1.4 1.1-.8l1.1 1.4 3.4-3.8c.6-.63 1.6-.27 1.2.7l-4 4.6c-.43.5-.8.4-1.1.1z'/%3e%3c/svg%3e") #fff no-repeat center right 1.75rem/calc(0.75em + 0.375rem) calc(0.75em + 0.375rem);
}
.was-validated .custom-select:valid:focus, .custom-select.is-valid:focus {
  border-color: #28a745;
  box-shadow: 0 0 0 0.2rem rgba(40, 167, 69, 0.25);
}

.was-validated .form-check-input:valid ~ .form-check-label, .form-check-input.is-valid ~ .form-check-label {
  color: #28a745;
}
.was-validated .form-check-input:valid ~ .valid-feedback,
.was-validated .form-check-input:valid ~ .valid-tooltip, .form-check-input.is-valid ~ .valid-feedback,
.form-check-input.is-valid ~ .valid-tooltip {
  display: block;
}

.was-validated .custom-control-input:valid ~ .custom-control-label, .custom-control-input.is-valid ~ .custom-control-label {
  color: #28a745;
}
.was-validated .custom-control-input:valid ~ .custom-control-label::before, .custom-control-input.is-valid ~ .custom-control-label::before {
  border-color: #28a745;
}
.was-validated .custom-control-input:valid:checked ~ .custom-control-label::before, .custom-control-input.is-valid:checked ~ .custom-control-label::before {
  border-color: #34ce57;
  background-color: #34ce57;
}
.was-validated .custom-control-input:valid:focus ~ .custom-control-label::before, .custom-control-input.is-valid:focus ~ .custom-control-label::before {
  box-shadow: 0 0 0 0.2rem rgba(40, 167, 69, 0.25);
}
.was-validated .custom-control-input:valid:focus:not(:checked) ~ .custom-control-label::before, .custom-control-input.is-valid:focus:not(:checked) ~ .custom-control-label::before {
  border-color: #28a745;
}

.was-validated .custom-file-input:valid ~ .custom-file-label, .custom-file-input.is-valid ~ .custom-file-label {
  border-color: #28a745;
}
.was-validated .custom-file-input:valid:focus ~ .custom-file-label, .custom-file-input.is-valid:focus ~ .custom-file-label {
  border-color: #28a745;
  box-shadow: 0 0 0 0.2rem rgba(40, 167, 69, 0.25);
}

.invalid-feedback {
  display: none;
  width: 100%;
  margin-top: 0.25rem;
  font-size: 80%;
  color: #dc3545;
}

.invalid-tooltip {
  position: absolute;
  top: 100%;
  left: 0;
  z-index: 5;
  display: none;
  max-width: 100%;
  padding: 0.25rem 0.5rem;
  margin-top: 0.1rem;
  font-size: 0.875rem;
  line-height: 1.5;
  color: #fff;
  background-color: rgba(220, 53, 69, 0.9);
  border-radius: 0.25rem;
}

.was-validated :invalid ~ .invalid-feedback,
.was-validated :invalid ~ .invalid-tooltip,
.is-invalid ~ .invalid-feedback,
.is-invalid ~ .invalid-tooltip {
  display: block;
}

.was-validated .form-control:invalid, .form-control.is-invalid {
  border-color: #dc3545;
  padding-right: calc(1.5em + 0.75rem);
  background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' fill='none' stroke='%23dc3545' viewBox='0 0 12 12'%3e%3ccircle cx='6' cy='6' r='4.5'/%3e%3cpath stroke-linejoin='round' d='M5.8 3.6h.4L6 6.5z'/%3e%3ccircle cx='6' cy='8.2' r='.6' fill='%23dc3545' stroke='none'/%3e%3c/svg%3e");
  background-repeat: no-repeat;
  background-position: right calc(0.375em + 0.1875rem) center;
  background-size: calc(0.75em + 0.375rem) calc(0.75em + 0.375rem);
}
.was-validated .form-control:invalid:focus, .form-control.is-invalid:focus {
  border-color: #dc3545;
  box-shadow: 0 0 0 0.2rem rgba(220, 53, 69, 0.25);
}

.was-validated textarea.form-control:invalid, textarea.form-control.is-invalid {
  padding-right: calc(1.5em + 0.75rem);
  background-position: top calc(0.375em + 0.1875rem) right calc(0.375em + 0.1875rem);
}

.was-validated .custom-select:invalid, .custom-select.is-invalid {
  border-color: #dc3545;
  padding-right: calc(0.75em + 2.3125rem);
  background: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='4' height='5' viewBox='0 0 4 5'%3e%3cpath fill='%23343a40' d='M2 0L0 2h4zm0 5L0 3h4z'/%3e%3c/svg%3e") no-repeat right 0.75rem center/8px 10px, url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' fill='none' stroke='%23dc3545' viewBox='0 0 12 12'%3e%3ccircle cx='6' cy='6' r='4.5'/%3e%3cpath stroke-linejoin='round' d='M5.8 3.6h.4L6 6.5z'/%3e%3ccircle cx='6' cy='8.2' r='.6' fill='%23dc3545' stroke='none'/%3e%3c/svg%3e") #fff no-repeat center right 1.75rem/calc(0.75em + 0.375rem) calc(0.75em + 0.375rem);
}
.was-validated .custom-select:invalid:focus, .custom-select.is-invalid:focus {
  border-color: #dc3545;
  box-shadow: 0 0 0 0.2rem rgba(220, 53, 69, 0.25);
}

.was-validated .form-check-input:invalid ~ .form-check-label, .form-check-input.is-invalid ~ .form-check-label {
  color: #dc3545;
}
.was-validated .form-check-input:invalid ~ .invalid-feedback,
.was-validated .form-check-input:invalid ~ .invalid-tooltip, .form-check-input.is-invalid ~ .invalid-feedback,
.form-check-input.is-invalid ~ .invalid-tooltip {
  display: block;
}

.was-validated .custom-control-input:invalid ~ .custom-control-label, .custom-control-input.is-invalid ~ .custom-control-label {
  color: #dc3545;
}
.was-validated .custom-control-input:invalid ~ .custom-control-label::before, .custom-control-input.is-invalid ~ .custom-control-label::before {
  border-color: #dc3545;
}
.was-validated .custom-control-input:invalid:checked ~ .custom-control-label::before, .custom-control-input.is-invalid:checked ~ .custom-control-label::before {
  border-color: #e4606d;
  background-color: #e4606d;
}
.was-validated .custom-control-input:invalid:focus ~ .custom-control-label::before, .custom-control-input.is-invalid:focus ~ .custom-control-label::before {
  box-shadow: 0 0 0 0.2rem rgba(220, 53, 69, 0.25);
}
.was-validated .custom-control-input:invalid:focus:not(:checked) ~ .custom-control-label::before, .custom-control-input.is-invalid:focus:not(:checked) ~ .custom-control-label::before {
  border-color: #dc3545;
}

.was-validated .custom-file-input:invalid ~ .custom-file-label, .custom-file-input.is-invalid ~ .custom-file-label {
  border-color: #dc3545;
}
.was-validated .custom-file-input:invalid:focus ~ .custom-file-label, .custom-file-input.is-invalid:focus ~ .custom-file-label {
  border-color: #dc3545;
  box-shadow: 0 0 0 0.2rem rgba(220, 53, 69, 0.25);
}

.form-inline {
  display: flex;
  flex-flow: row wrap;
  align-items: center;
}
.form-inline .form-check {
  width: 100%;
}
@media (min-width: 576px) {
  .form-inline label {
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 0;
  }
  .form-inline .form-group {
    display: flex;
    flex: 0 0 auto;
    flex-flow: row wrap;
    align-items: center;
    margin-bottom: 0;
  }
  .form-inline .form-control {
    display: inline-block;
    width: auto;
    vertical-align: middle;
  }
  .form-inline .form-control-plaintext {
    display: inline-block;
  }
  .form-inline .input-group,
.form-inline .custom-select {
    width: auto;
  }
  .form-inline .form-check {
    display: flex;
    align-items: center;
    justify-content: center;
    width: auto;
    padding-left: 0;
  }
  .form-inline .form-check-input {
    position: relative;
    flex-shrink: 0;
    margin-top: 0;
    margin-right: 0.25rem;
    margin-left: 0;
  }
  .form-inline .custom-control {
    align-items: center;
    justify-content: center;
  }
  .form-inline .custom-control-label {
    margin-bottom: 0;
  }
}

.btn {
  display: inline-block;
  font-weight: 400;
  color: #212529;
  text-align: center;
  vertical-align: middle;
  -webkit-user-select: none;
     -moz-user-select: none;
      -ms-user-select: none;
          user-select: none;
  background-color: transparent;
  border: 1px solid transparent;
  padding: 0.375rem 0.75rem;
  font-size: 1rem;
  line-height: 1.5;
  border-radius: 0.25rem;
  transition: color 0.15s ease-in-out, background-color 0.15s ease-in-out, border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
}
@media (prefers-reduced-motion: reduce) {
  .btn {
    transition: none;
  }
}
.btn:hover {
  color: #212529;
  text-decoration: none;
}
.btn:focus, .btn.focus {
  outline: 0;
  box-shadow: 0 0 0 0.2rem rgba(254, 209, 54, 0.25);
}
.btn.disabled, .btn:disabled {
  opacity: 0.65;
}
.btn:not(:disabled):not(.disabled) {
  cursor: pointer;
}
a.btn.disabled,
fieldset:disabled a.btn {
  pointer-events: none;
}

.btn-primary {
  color: #fff;
  background-color: #fed136;
  border-color: #fed136;
}
.btn-primary:hover {
  color: #fff;
  background-color: #fec810;
  border-color: #fec503;
}
.btn-primary:focus, .btn-primary.focus {
  color: #fff;
  background-color: #fec810;
  border-color: #fec503;
  box-shadow: 0 0 0 0.2rem rgba(254, 216, 84, 0.5);
}
.btn-primary.disabled, .btn-primary:disabled {
  color: #fff;
  background-color: #fed136;
  border-color: #fed136;
}
.btn-primary:not(:disabled):not(.disabled):active, .btn-primary:not(:disabled):not(.disabled).active, .show > .btn-primary.dropdown-toggle {
  color: #fff;
  background-color: #fec503;
  border-color: #f3bd01;
}
.btn-primary:not(:disabled):not(.disabled):active:focus, .btn-primary:not(:disabled):not(.disabled).active:focus, .show > .btn-primary.dropdown-toggle:focus {
  box-shadow: 0 0 0 0.2rem rgba(254, 216, 84, 0.5);
}

.btn-secondary {
  color: #fff;
  background-color: #6c757d;
  border-color: #6c757d;
}
.btn-secondary:hover {
  color: #fff;
  background-color: #5a6268;
  border-color: #545b62;
}
.btn-secondary:focus, .btn-secondary.focus {
  color: #fff;
  background-color: #5a6268;
  border-color: #545b62;
  box-shadow: 0 0 0 0.2rem rgba(130, 138, 145, 0.5);
}
.btn-secondary.disabled, .btn-secondary:disabled {
  color: #fff;
  background-color: #6c757d;
  border-color: #6c757d;
}
.btn-secondary:not(:disabled):not(.disabled):active, .btn-secondary:not(:disabled):not(.disabled).active, .show > .btn-secondary.dropdown-toggle {
  color: #fff;
  background-color: #545b62;
  border-color: #4e555b;
}
.btn-secondary:not(:disabled):not(.disabled):active:focus, .btn-secondary:not(:disabled):not(.disabled).active:focus, .show > .btn-secondary.dropdown-toggle:focus {
  box-shadow: 0 0 0 0.2rem rgba(130, 138, 145, 0.5);
}

.btn-success {
  color: #fff;
  background-color: #28a745;
  border-color: #28a745;
}
.btn-success:hover {
  color: #fff;
  background-color: #218838;
  border-color: #1e7e34;
}
.btn-success:focus, .btn-success.focus {
  color: #fff;
  background-color: #218838;
  border-color: #1e7e34;
  box-shadow: 0 0 0 0.2rem rgba(72, 180, 97, 0.5);
}
.btn-success.disabled, .btn-success:disabled {
  color: #fff;
  background-color: #28a745;
  border-color: #28a745;
}
.btn-success:not(:disabled):not(.disabled):active, .btn-success:not(:disabled):not(.disabled).active, .show > .btn-success.dropdown-toggle {
  color: #fff;
  background-color: #1e7e34;
  border-color: #1c7430;
}
.btn-success:not(:disabled):not(.disabled):active:focus, .btn-success:not(:disabled):not(.disabled).active:focus, .show > .btn-success.dropdown-toggle:focus {
  box-shadow: 0 0 0 0.2rem rgba(72, 180, 97, 0.5);
}

.btn-info {
  color: #fff;
  background-color: #17a2b8;
  border-color: #17a2b8;
}
.btn-info:hover {
  color: #fff;
  background-color: #138496;
  border-color: #117a8b;
}
.btn-info:focus, .btn-info.focus {
  color: #fff;
  background-color: #138496;
  border-color: #117a8b;
  box-shadow: 0 0 0 0.2rem rgba(58, 176, 195, 0.5);
}
.btn-info.disabled, .btn-info:disabled {
  color: #fff;
  background-color: #17a2b8;
  border-color: #17a2b8;
}
.btn-info:not(:disabled):not(.disabled):active, .btn-info:not(:disabled):not(.disabled).active, .show > .btn-info.dropdown-toggle {
  color: #fff;
  background-color: #117a8b;
  border-color: #10707f;
}
.btn-info:not(:disabled):not(.disabled):active:focus, .btn-info:not(:disabled):not(.disabled).active:focus, .show > .btn-info.dropdown-toggle:focus {
  box-shadow: 0 0 0 0.2rem rgba(58, 176, 195, 0.5);
}

.btn-warning {
  color: #fff;
  background-color: #fed136;
  border-color: #fed136;
}
.btn-warning:hover {
  color: #fff;
  background-color: #fec810;
  border-color: #fec503;
}
.btn-warning:focus, .btn-warning.focus {
  color: #fff;
  background-color: #fec810;
  border-color: #fec503;
  box-shadow: 0 0 0 0.2rem rgba(254, 216, 84, 0.5);
}
.btn-warning.disabled, .btn-warning:disabled {
  color: #fff;
  background-color: #fed136;
  border-color: #fed136;
}
.btn-warning:not(:disabled):not(.disabled):active, .btn-warning:not(:disabled):not(.disabled).active, .show > .btn-warning.dropdown-toggle {
  color: #fff;
  background-color: #fec503;
  border-color: #f3bd01;
}
.btn-warning:not(:disabled):not(.disabled):active:focus, .btn-warning:not(:disabled):not(.disabled).active:focus, .show > .btn-warning.dropdown-toggle:focus {
  box-shadow: 0 0 0 0.2rem rgba(254, 216, 84, 0.5);
}

.btn-danger {
  color: #fff;
  background-color: #dc3545;
  border-color: #dc3545;
}
.btn-danger:hover {
  color: #fff;
  background-color: #c82333;
  border-color: #bd2130;
}
.btn-danger:focus, .btn-danger.focus {
  color: #fff;
  background-color: #c82333;
  border-color: #bd2130;
  box-shadow: 0 0 0 0.2rem rgba(225, 83, 97, 0.5);
}
.btn-danger.disabled, .btn-danger:disabled {
  color: #fff;
  background-color: #dc3545;
  border-color: #dc3545;
}
.btn-danger:not(:disabled):not(.disabled):active, .btn-danger:not(:disabled):not(.disabled).active, .show > .btn-danger.dropdown-toggle {
  color: #fff;
  background-color: #bd2130;
  border-color: #b21f2d;
}
.btn-danger:not(:disabled):not(.disabled):active:focus, .btn-danger:not(:disabled):not(.disabled).active:focus, .show > .btn-danger.dropdown-toggle:focus {
  box-shadow: 0 0 0 0.2rem rgba(225, 83, 97, 0.5);
}

.btn-light {
  color: #212529;
  background-color: #f8f9fa;
  border-color: #f8f9fa;
}
.btn-light:hover {
  color: #212529;
  background-color: #e2e6ea;
  border-color: #dae0e5;
}
.btn-light:focus, .btn-light.focus {
  color: #212529;
  background-color: #e2e6ea;
  border-color: #dae0e5;
  box-shadow: 0 0 0 0.2rem rgba(216, 217, 219, 0.5);
}
.btn-light.disabled, .btn-light:disabled {
  color: #212529;
  background-color: #f8f9fa;
  border-color: #f8f9fa;
}
.btn-light:not(:disabled):not(.disabled):active, .btn-light:not(:disabled):not(.disabled).active, .show > .btn-light.dropdown-toggle {
  color: #212529;
  background-color: #dae0e5;
  border-color: #d3d9df;
}
.btn-light:not(:disabled):not(.disabled):active:focus, .btn-light:not(:disabled):not(.disabled).active:focus, .show > .btn-light.dropdown-toggle:focus {
  box-shadow: 0 0 0 0.2rem rgba(216, 217, 219, 0.5);
}

.btn-dark {
  color: #fff;
  background-color: #343a40;
  border-color: #343a40;
}
.btn-dark:hover {
  color: #fff;
  background-color: #23272b;
  border-color: #1d2124;
}
.btn-dark:focus, .btn-dark.focus {
  color: #fff;
  background-color: #23272b;
  border-color: #1d2124;
  box-shadow: 0 0 0 0.2rem rgba(82, 88, 93, 0.5);
}
.btn-dark.disabled, .btn-dark:disabled {
  color: #fff;
  background-color: #343a40;
  border-color: #343a40;
}
.btn-dark:not(:disabled):not(.disabled):active, .btn-dark:not(:disabled):not(.disabled).active, .show > .btn-dark.dropdown-toggle {
  color: #fff;
  background-color: #1d2124;
  border-color: #171a1d;
}
.btn-dark:not(:disabled):not(.disabled):active:focus, .btn-dark:not(:disabled):not(.disabled).active:focus, .show > .btn-dark.dropdown-toggle:focus {
  box-shadow: 0 0 0 0.2rem rgba(82, 88, 93, 0.5);
}

.btn-outline-primary {
  color: #fed136;
  border-color: #fed136;
}
.btn-outline-primary:hover {
  color: #fff;
  background-color: #fed136;
  border-color: #fed136;
}
.btn-outline-primary:focus, .btn-outline-primary.focus {
  box-shadow: 0 0 0 0.2rem rgba(254, 209, 54, 0.5);
}
.btn-outline-primary.disabled, .btn-outline-primary:disabled {
  color: #fed136;
  background-color: transparent;
}
.btn-outline-primary:not(:disabled):not(.disabled):active, .btn-outline-primary:not(:disabled):not(.disabled).active, .show > .btn-outline-primary.dropdown-toggle {
  color: #fff;
  background-color: #fed136;
  border-color: #fed136;
}
.btn-outline-primary:not(:disabled):not(.disabled):active:focus, .btn-outline-primary:not(:disabled):not(.disabled).active:focus, .show > .btn-outline-primary.dropdown-toggle:focus {
  box-shadow: 0 0 0 0.2rem rgba(254, 209, 54, 0.5);
}

.btn-outline-secondary {
  color: #6c757d;
  border-color: #6c757d;
}
.btn-outline-secondary:hover {
  color: #fff;
  background-color: #6c757d;
  border-color: #6c757d;
}
.btn-outline-secondary:focus, .btn-outline-secondary.focus {
  box-shadow: 0 0 0 0.2rem rgba(108, 117, 125, 0.5);
}
.btn-outline-secondary.disabled, .btn-outline-secondary:disabled {
  color: #6c757d;
  background-color: transparent;
}
.btn-outline-secondary:not(:disabled):not(.disabled):active, .btn-outline-secondary:not(:disabled):not(.disabled).active, .show > .btn-outline-secondary.dropdown-toggle {
  color: #fff;
  background-color: #6c757d;
  border-color: #6c757d;
}
.btn-outline-secondary:not(:disabled):not(.disabled):active:focus, .btn-outline-secondary:not(:disabled):not(.disabled).active:focus, .show > .btn-outline-secondary.dropdown-toggle:focus {
  box-shadow: 0 0 0 0.2rem rgba(108, 117, 125, 0.5);
}

.btn-outline-success {
  color: #28a745;
  border-color: #28a745;
}
.btn-outline-success:hover {
  color: #fff;
  background-color: #28a745;
  border-color: #28a745;
}
.btn-outline-success:focus, .btn-outline-success.focus {
  box-shadow: 0 0 0 0.2rem rgba(40, 167, 69, 0.5);
}
.btn-outline-success.disabled, .btn-outline-success:disabled {
  color: #28a745;
  background-color: transparent;
}
.btn-outline-success:not(:disabled):not(.disabled):active, .btn-outline-success:not(:disabled):not(.disabled).active, .show > .btn-outline-success.dropdown-toggle {
  color: #fff;
  background-color: #28a745;
  border-color: #28a745;
}
.btn-outline-success:not(:disabled):not(.disabled):active:focus, .btn-outline-success:not(:disabled):not(.disabled).active:focus, .show > .btn-outline-success.dropdown-toggle:focus {
  box-shadow: 0 0 0 0.2rem rgba(40, 167, 69, 0.5);
}

.btn-outline-info {
  color: #17a2b8;
  border-color: #17a2b8;
}
.btn-outline-info:hover {
  color: #fff;
  background-color: #17a2b8;
  border-color: #17a2b8;
}
.btn-outline-info:focus, .btn-outline-info.focus {
  box-shadow: 0 0 0 0.2rem rgba(23, 162, 184, 0.5);
}
.btn-outline-info.disabled, .btn-outline-info:disabled {
  color: #17a2b8;
  background-color: transparent;
}
.btn-outline-info:not(:disabled):not(.disabled):active, .btn-outline-info:not(:disabled):not(.disabled).active, .show > .btn-outline-info.dropdown-toggle {
  color: #fff;
  background-color: #17a2b8;
  border-color: #17a2b8;
}
.btn-outline-info:not(:disabled):not(.disabled):active:focus, .btn-outline-info:not(:disabled):not(.disabled).active:focus, .show > .btn-outline-info.dropdown-toggle:focus {
  box-shadow: 0 0 0 0.2rem rgba(23, 162, 184, 0.5);
}

.btn-outline-warning {
  color: #fed136;
  border-color: #fed136;
}
.btn-outline-warning:hover {
  color: #fff;
  background-color: #fed136;
  border-color: #fed136;
}
.btn-outline-warning:focus, .btn-outline-warning.focus {
  box-shadow: 0 0 0 0.2rem rgba(254, 209, 54, 0.5);
}
.btn-outline-warning.disabled, .btn-outline-warning:disabled {
  color: #fed136;
  background-color: transparent;
}
.btn-outline-warning:not(:disabled):not(.disabled):active, .btn-outline-warning:not(:disabled):not(.disabled).active, .show > .btn-outline-warning.dropdown-toggle {
  color: #fff;
  background-color: #fed136;
  border-color: #fed136;
}
.btn-outline-warning:not(:disabled):not(.disabled):active:focus, .btn-outline-warning:not(:disabled):not(.disabled).active:focus, .show > .btn-outline-warning.dropdown-toggle:focus {
  box-shadow: 0 0 0 0.2rem rgba(254, 209, 54, 0.5);
}

.btn-outline-danger {
  color: #dc3545;
  border-color: #dc3545;
}
.btn-outline-danger:hover {
  color: #fff;
  background-color: #dc3545;
  border-color: #dc3545;
}
.btn-outline-danger:focus, .btn-outline-danger.focus {
  box-shadow: 0 0 0 0.2rem rgba(220, 53, 69, 0.5);
}
.btn-outline-danger.disabled, .btn-outline-danger:disabled {
  color: #dc3545;
  background-color: transparent;
}
.btn-outline-danger:not(:disabled):not(.disabled):active, .btn-outline-danger:not(:disabled):not(.disabled).active, .show > .btn-outline-danger.dropdown-toggle {
  color: #fff;
  background-color: #dc3545;
  border-color: #dc3545;
}
.btn-outline-danger:not(:disabled):not(.disabled):active:focus, .btn-outline-danger:not(:disabled):not(.disabled).active:focus, .show > .btn-outline-danger.dropdown-toggle:focus {
  box-shadow: 0 0 0 0.2rem rgba(220, 53, 69, 0.5);
}

.btn-outline-light {
  color: #f8f9fa;
  border-color: #f8f9fa;
}
.btn-outline-light:hover {
  color: #212529;
  background-color: #f8f9fa;
  border-color: #f8f9fa;
}
.btn-outline-light:focus, .btn-outline-light.focus {
  box-shadow: 0 0 0 0.2rem rgba(248, 249, 250, 0.5);
}
.btn-outline-light.disabled, .btn-outline-light:disabled {
  color: #f8f9fa;
  background-color: transparent;
}
.btn-outline-light:not(:disabled):not(.disabled):active, .btn-outline-light:not(:disabled):not(.disabled).active, .show > .btn-outline-light.dropdown-toggle {
  color: #212529;
  background-color: #f8f9fa;
  border-color: #f8f9fa;
}
.btn-outline-light:not(:disabled):not(.disabled):active:focus, .btn-outline-light:not(:disabled):not(.disabled).active:focus, .show > .btn-outline-light.dropdown-toggle:focus {
  box-shadow: 0 0 0 0.2rem rgba(248, 249, 250, 0.5);
}

.btn-outline-dark {
  color: #343a40;
  border-color: #343a40;
}
.btn-outline-dark:hover {
  color: #fff;
  background-color: #343a40;
  border-color: #343a40;
}
.btn-outline-dark:focus, .btn-outline-dark.focus {
  box-shadow: 0 0 0 0.2rem rgba(52, 58, 64, 0.5);
}
.btn-outline-dark.disabled, .btn-outline-dark:disabled {
  color: #343a40;
  background-color: transparent;
}
.btn-outline-dark:not(:disabled):not(.disabled):active, .btn-outline-dark:not(:disabled):not(.disabled).active, .show > .btn-outline-dark.dropdown-toggle {
  color: #fff;
  background-color: #343a40;
  border-color: #343a40;
}
.btn-outline-dark:not(:disabled):not(.disabled):active:focus, .btn-outline-dark:not(:disabled):not(.disabled).active:focus, .show > .btn-outline-dark.dropdown-toggle:focus {
  box-shadow: 0 0 0 0.2rem rgba(52, 58, 64, 0.5);
}

.btn-link {
  font-weight: 400;
  color: #fed136;
  text-decoration: none;
}
.btn-link:hover {
  color: #e6b301;
  text-decoration: underline;
}
.btn-link:focus, .btn-link.focus {
  text-decoration: underline;
}
.btn-link:disabled, .btn-link.disabled {
  color: #6c757d;
  pointer-events: none;
}

.btn-lg, .btn-group-lg > .btn {
  padding: 0.5rem 1rem;
  font-size: 1.25rem;
  line-height: 1.5;
  border-radius: 0.3rem;
}

.btn-sm, .btn-group-sm > .btn {
  padding: 0.25rem 0.5rem;
  font-size: 0.875rem;
  line-height: 1.5;
  border-radius: 0.2rem;
}

.btn-block {
  display: block;
  width: 100%;
}
.btn-block + .btn-block {
  margin-top: 0.5rem;
}

input[type=submit].btn-block,
input[type=reset].btn-block,
input[type=button].btn-block {
  width: 100%;
}

.fade {
  transition: opacity 0.15s linear;
}
@media (prefers-reduced-motion: reduce) {
  .fade {
    transition: none;
  }
}
.fade:not(.show) {
  opacity: 0;
}

.collapse:not(.show) {
  display: none;
}

.collapsing {
  position: relative;
  height: 0;
  overflow: hidden;
  transition: height 0.35s ease;
}
@media (prefers-reduced-motion: reduce) {
  .collapsing {
    transition: none;
  }
}

.dropup,
.dropright,
.dropdown,
.dropleft {
  position: relative;
}

.dropdown-toggle {
  white-space: nowrap;
}
.dropdown-toggle::after {
  display: inline-block;
  margin-left: 0.255em;
  vertical-align: 0.255em;
  content: "";
  border-top: 0.3em solid;
  border-right: 0.3em solid transparent;
  border-bottom: 0;
  border-left: 0.3em solid transparent;
}
.dropdown-toggle:empty::after {
  margin-left: 0;
}

.dropdown-menu {
  position: absolute;
  top: 100%;
  left: 0;
  z-index: 1000;
  display: none;
  float: left;
  min-width: 10rem;
  padding: 0.5rem 0;
  margin: 0.125rem 0 0;
  font-size: 1rem;
  color: #212529;
  text-align: left;
  list-style: none;
  background-color: #fff;
  background-clip: padding-box;
  border: 1px solid rgba(0, 0, 0, 0.15);
  border-radius: 0.25rem;
}

.dropdown-menu-left {
  right: auto;
  left: 0;
}

.dropdown-menu-right {
  right: 0;
  left: auto;
}

@media (min-width: 576px) {
  .dropdown-menu-sm-left {
    right: auto;
    left: 0;
  }

  .dropdown-menu-sm-right {
    right: 0;
    left: auto;
  }
}
@media (min-width: 768px) {
  .dropdown-menu-md-left {
    right: auto;
    left: 0;
  }

  .dropdown-menu-md-right {
    right: 0;
    left: auto;
  }
}
@media (min-width: 992px) {
  .dropdown-menu-lg-left {
    right: auto;
    left: 0;
  }

  .dropdown-menu-lg-right {
    right: 0;
    left: auto;
  }
}
@media (min-width: 1200px) {
  .dropdown-menu-xl-left {
    right: auto;
    left: 0;
  }

  .dropdown-menu-xl-right {
    right: 0;
    left: auto;
  }
}
.dropup .dropdown-menu {
  top: auto;
  bottom: 100%;
  margin-top: 0;
  margin-bottom: 0.125rem;
}
.dropup .dropdown-toggle::after {
  display: inline-block;
  margin-left: 0.255em;
  vertical-align: 0.255em;
  content: "";
  border-top: 0;
  border-right: 0.3em solid transparent;
  border-bottom: 0.3em solid;
  border-left: 0.3em solid transparent;
}
.dropup .dropdown-toggle:empty::after {
  margin-left: 0;
}

.dropright .dropdown-menu {
  top: 0;
  right: auto;
  left: 100%;
  margin-top: 0;
  margin-left: 0.125rem;
}
.dropright .dropdown-toggle::after {
  display: inline-block;
  margin-left: 0.255em;
  vertical-align: 0.255em;
  content: "";
  border-top: 0.3em solid transparent;
  border-right: 0;
  border-bottom: 0.3em solid transparent;
  border-left: 0.3em solid;
}
.dropright .dropdown-toggle:empty::after {
  margin-left: 0;
}
.dropright .dropdown-toggle::after {
  vertical-align: 0;
}

.dropleft .dropdown-menu {
  top: 0;
  right: 100%;
  left: auto;
  margin-top: 0;
  margin-right: 0.125rem;
}
.dropleft .dropdown-toggle::after {
  display: inline-block;
  margin-left: 0.255em;
  vertical-align: 0.255em;
  content: "";
}
.dropleft .dropdown-toggle::after {
  display: none;
}
.dropleft .dropdown-toggle::before {
  display: inline-block;
  margin-right: 0.255em;
  vertical-align: 0.255em;
  content: "";
  border-top: 0.3em solid transparent;
  border-right: 0.3em solid;
  border-bottom: 0.3em solid transparent;
}
.dropleft .dropdown-toggle:empty::after {
  margin-left: 0;
}
.dropleft .dropdown-toggle::before {
  vertical-align: 0;
}

.dropdown-menu[x-placement^=top], .dropdown-menu[x-placement^=right], .dropdown-menu[x-placement^=bottom], .dropdown-menu[x-placement^=left] {
  right: auto;
  bottom: auto;
}

.dropdown-divider {
  height: 0;
  margin: 0.5rem 0;
  overflow: hidden;
  border-top: 1px solid #e9ecef;
}

.dropdown-item {
  display: block;
  width: 100%;
  padding: 0.25rem 1.5rem;
  clear: both;
  font-weight: 400;
  color: #212529;
  text-align: inherit;
  white-space: nowrap;
  background-color: transparent;
  border: 0;
}
.dropdown-item:hover, .dropdown-item:focus {
  color: #16181b;
  text-decoration: none;
  background-color: #f8f9fa;
}
.dropdown-item.active, .dropdown-item:active {
  color: #fff;
  text-decoration: none;
  background-color: #fed136;
}
.dropdown-item.disabled, .dropdown-item:disabled {
  color: #6c757d;
  pointer-events: none;
  background-color: transparent;
}

.dropdown-menu.show {
  display: block;
}

.dropdown-header {
  display: block;
  padding: 0.5rem 1.5rem;
  margin-bottom: 0;
  font-size: 0.875rem;
  color: #6c757d;
  white-space: nowrap;
}

.dropdown-item-text {
  display: block;
  padding: 0.25rem 1.5rem;
  color: #212529;
}

.btn-group,
.btn-group-vertical {
  position: relative;
  display: inline-flex;
  vertical-align: middle;
}
.btn-group > .btn,
.btn-group-vertical > .btn {
  position: relative;
  flex: 1 1 auto;
}
.btn-group > .btn:hover,
.btn-group-vertical > .btn:hover {
  z-index: 1;
}
.btn-group > .btn:focus, .btn-group > .btn:active, .btn-group > .btn.active,
.btn-group-vertical > .btn:focus,
.btn-group-vertical > .btn:active,
.btn-group-vertical > .btn.active {
  z-index: 1;
}

.btn-toolbar {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-start;
}
.btn-toolbar .input-group {
  width: auto;
}

.btn-group > .btn:not(:first-child),
.btn-group > .btn-group:not(:first-child) {
  margin-left: -1px;
}
.btn-group > .btn:not(:last-child):not(.dropdown-toggle),
.btn-group > .btn-group:not(:last-child) > .btn {
  border-top-right-radius: 0;
  border-bottom-right-radius: 0;
}
.btn-group > .btn:not(:first-child),
.btn-group > .btn-group:not(:first-child) > .btn {
  border-top-left-radius: 0;
  border-bottom-left-radius: 0;
}

.dropdown-toggle-split {
  padding-right: 0.5625rem;
  padding-left: 0.5625rem;
}
.dropdown-toggle-split::after, .dropup .dropdown-toggle-split::after, .dropright .dropdown-toggle-split::after {
  margin-left: 0;
}
.dropleft .dropdown-toggle-split::before {
  margin-right: 0;
}

.btn-sm + .dropdown-toggle-split, .btn-group-sm > .btn + .dropdown-toggle-split {
  padding-right: 0.375rem;
  padding-left: 0.375rem;
}

.btn-lg + .dropdown-toggle-split, .btn-group-lg > .btn + .dropdown-toggle-split {
  padding-right: 0.75rem;
  padding-left: 0.75rem;
}

.btn-group-vertical {
  flex-direction: column;
  align-items: flex-start;
  justify-content: center;
}
.btn-group-vertical > .btn,
.btn-group-vertical > .btn-group {
  width: 100%;
}
.btn-group-vertical > .btn:not(:first-child),
.btn-group-vertical > .btn-group:not(:first-child) {
  margin-top: -1px;
}
.btn-group-vertical > .btn:not(:last-child):not(.dropdown-toggle),
.btn-group-vertical > .btn-group:not(:last-child) > .btn {
  border-bottom-right-radius: 0;
  border-bottom-left-radius: 0;
}
.btn-group-vertical > .btn:not(:first-child),
.btn-group-vertical > .btn-group:not(:first-child) > .btn {
  border-top-left-radius: 0;
  border-top-right-radius: 0;
}

.btn-group-toggle > .btn,
.btn-group-toggle > .btn-group > .btn {
  margin-bottom: 0;
}
.btn-group-toggle > .btn input[type=radio],
.btn-group-toggle > .btn input[type=checkbox],
.btn-group-toggle > .btn-group > .btn input[type=radio],
.btn-group-toggle > .btn-group > .btn input[type=checkbox] {
  position: absolute;
  clip: rect(0, 0, 0, 0);
  pointer-events: none;
}

.input-group {
  position: relative;
  display: flex;
  flex-wrap: wrap;
  align-items: stretch;
  width: 100%;
}
.input-group > .form-control,
.input-group > .form-control-plaintext,
.input-group > .custom-select,
.input-group > .custom-file {
  position: relative;
  flex: 1 1 auto;
  width: 1%;
  min-width: 0;
  margin-bottom: 0;
}
.input-group > .form-control + .form-control,
.input-group > .form-control + .custom-select,
.input-group > .form-control + .custom-file,
.input-group > .form-control-plaintext + .form-control,
.input-group > .form-control-plaintext + .custom-select,
.input-group > .form-control-plaintext + .custom-file,
.input-group > .custom-select + .form-control,
.input-group > .custom-select + .custom-select,
.input-group > .custom-select + .custom-file,
.input-group > .custom-file + .form-control,
.input-group > .custom-file + .custom-select,
.input-group > .custom-file + .custom-file {
  margin-left: -1px;
}
.input-group > .form-control:focus,
.input-group > .custom-select:focus,
.input-group > .custom-file .custom-file-input:focus ~ .custom-file-label {
  z-index: 3;
}
.input-group > .custom-file .custom-file-input:focus {
  z-index: 4;
}
.input-group > .form-control:not(:last-child),
.input-group > .custom-select:not(:last-child) {
  border-top-right-radius: 0;
  border-bottom-right-radius: 0;
}
.input-group > .form-control:not(:first-child),
.input-group > .custom-select:not(:first-child) {
  border-top-left-radius: 0;
  border-bottom-left-radius: 0;
}
.input-group > .custom-file {
  display: flex;
  align-items: center;
}
.input-group > .custom-file:not(:last-child) .custom-file-label, .input-group > .custom-file:not(:last-child) .custom-file-label::after {
  border-top-right-radius: 0;
  border-bottom-right-radius: 0;
}
.input-group > .custom-file:not(:first-child) .custom-file-label {
  border-top-left-radius: 0;
  border-bottom-left-radius: 0;
}

.input-group-prepend,
.input-group-append {
  display: flex;
}
.input-group-prepend .btn,
.input-group-append .btn {
  position: relative;
  z-index: 2;
}
.input-group-prepend .btn:focus,
.input-group-append .btn:focus {
  z-index: 3;
}
.input-group-prepend .btn + .btn,
.input-group-prepend .btn + .input-group-text,
.input-group-prepend .input-group-text + .input-group-text,
.input-group-prepend .input-group-text + .btn,
.input-group-append .btn + .btn,
.input-group-append .btn + .input-group-text,
.input-group-append .input-group-text + .input-group-text,
.input-group-append .input-group-text + .btn {
  margin-left: -1px;
}

.input-group-prepend {
  margin-right: -1px;
}

.input-group-append {
  margin-left: -1px;
}

.input-group-text {
  display: flex;
  align-items: center;
  padding: 0.375rem 0.75rem;
  margin-bottom: 0;
  font-size: 1rem;
  font-weight: 400;
  line-height: 1.5;
  color: #495057;
  text-align: center;
  white-space: nowrap;
  background-color: #e9ecef;
  border: 1px solid #ced4da;
  border-radius: 0.25rem;
}
.input-group-text input[type=radio],
.input-group-text input[type=checkbox] {
  margin-top: 0;
}

.input-group-lg > .form-control:not(textarea),
.input-group-lg > .custom-select {
  height: calc(1.5em + 1rem + 2px);
}

.input-group-lg > .form-control,
.input-group-lg > .custom-select,
.input-group-lg > .input-group-prepend > .input-group-text,
.input-group-lg > .input-group-append > .input-group-text,
.input-group-lg > .input-group-prepend > .btn,
.input-group-lg > .input-group-append > .btn {
  padding: 0.5rem 1rem;
  font-size: 1.25rem;
  line-height: 1.5;
  border-radius: 0.3rem;
}

.input-group-sm > .form-control:not(textarea),
.input-group-sm > .custom-select {
  height: calc(1.5em + 0.5rem + 2px);
}

.input-group-sm > .form-control,
.input-group-sm > .custom-select,
.input-group-sm > .input-group-prepend > .input-group-text,
.input-group-sm > .input-group-append > .input-group-text,
.input-group-sm > .input-group-prepend > .btn,
.input-group-sm > .input-group-append > .btn {
  padding: 0.25rem 0.5rem;
  font-size: 0.875rem;
  line-height: 1.5;
  border-radius: 0.2rem;
}

.input-group-lg > .custom-select,
.input-group-sm > .custom-select {
  padding-right: 1.75rem;
}

.input-group > .input-group-prepend > .btn,
.input-group > .input-group-prepend > .input-group-text,
.input-group > .input-group-append:not(:last-child) > .btn,
.input-group > .input-group-append:not(:last-child) > .input-group-text,
.input-group > .input-group-append:last-child > .btn:not(:last-child):not(.dropdown-toggle),
.input-group > .input-group-append:last-child > .input-group-text:not(:last-child) {
  border-top-right-radius: 0;
  border-bottom-right-radius: 0;
}

.input-group > .input-group-append > .btn,
.input-group > .input-group-append > .input-group-text,
.input-group > .input-group-prepend:not(:first-child) > .btn,
.input-group > .input-group-prepend:not(:first-child) > .input-group-text,
.input-group > .input-group-prepend:first-child > .btn:not(:first-child),
.input-group > .input-group-prepend:first-child > .input-group-text:not(:first-child) {
  border-top-left-radius: 0;
  border-bottom-left-radius: 0;
}

.custom-control {
  position: relative;
  z-index: 1;
  display: block;
  min-height: 1.5rem;
  padding-left: 1.5rem;
  -webkit-print-color-adjust: exact;
          color-adjust: exact;
}

.custom-control-inline {
  display: inline-flex;
  margin-right: 1rem;
}

.custom-control-input {
  position: absolute;
  left: 0;
  z-index: -1;
  width: 1rem;
  height: 1.25rem;
  opacity: 0;
}
.custom-control-input:checked ~ .custom-control-label::before {
  color: #fff;
  border-color: #fed136;
  background-color: #fed136;
}
.custom-control-input:focus ~ .custom-control-label::before {
  box-shadow: 0 0 0 0.2rem rgba(254, 209, 54, 0.25);
}
.custom-control-input:focus:not(:checked) ~ .custom-control-label::before {
  border-color: #ffeeb5;
}
.custom-control-input:not(:disabled):active ~ .custom-control-label::before {
  color: #fff;
  background-color: #fffae8;
  border-color: #fffae8;
}
.custom-control-input[disabled] ~ .custom-control-label, .custom-control-input:disabled ~ .custom-control-label {
  color: #6c757d;
}
.custom-control-input[disabled] ~ .custom-control-label::before, .custom-control-input:disabled ~ .custom-control-label::before {
  background-color: #e9ecef;
}

.custom-control-label {
  position: relative;
  margin-bottom: 0;
  vertical-align: top;
}
.custom-control-label::before {
  position: absolute;
  top: 0.25rem;
  left: -1.5rem;
  display: block;
  width: 1rem;
  height: 1rem;
  pointer-events: none;
  content: "";
  background-color: #fff;
  border: #adb5bd solid 1px;
}
.custom-control-label::after {
  position: absolute;
  top: 0.25rem;
  left: -1.5rem;
  display: block;
  width: 1rem;
  height: 1rem;
  content: "";
  background: no-repeat 50%/50% 50%;
}

.custom-checkbox .custom-control-label::before {
  border-radius: 0.25rem;
}
.custom-checkbox .custom-control-input:checked ~ .custom-control-label::after {
  background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='8' height='8' viewBox='0 0 8 8'%3e%3cpath fill='%23fff' d='M6.564.75l-3.59 3.612-1.538-1.55L0 4.26l2.974 2.99L8 2.193z'/%3e%3c/svg%3e");
}
.custom-checkbox .custom-control-input:indeterminate ~ .custom-control-label::before {
  border-color: #fed136;
  background-color: #fed136;
}
.custom-checkbox .custom-control-input:indeterminate ~ .custom-control-label::after {
  background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='4' height='4' viewBox='0 0 4 4'%3e%3cpath stroke='%23fff' d='M0 2h4'/%3e%3c/svg%3e");
}
.custom-checkbox .custom-control-input:disabled:checked ~ .custom-control-label::before {
  background-color: rgba(254, 209, 54, 0.5);
}
.custom-checkbox .custom-control-input:disabled:indeterminate ~ .custom-control-label::before {
  background-color: rgba(254, 209, 54, 0.5);
}

.custom-radio .custom-control-label::before {
  border-radius: 50%;
}
.custom-radio .custom-control-input:checked ~ .custom-control-label::after {
  background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='-4 -4 8 8'%3e%3ccircle r='3' fill='%23fff'/%3e%3c/svg%3e");
}
.custom-radio .custom-control-input:disabled:checked ~ .custom-control-label::before {
  background-color: rgba(254, 209, 54, 0.5);
}

.custom-switch {
  padding-left: 2.25rem;
}
.custom-switch .custom-control-label::before {
  left: -2.25rem;
  width: 1.75rem;
  pointer-events: all;
  border-radius: 0.5rem;
}
.custom-switch .custom-control-label::after {
  top: calc(0.25rem + 2px);
  left: calc(-2.25rem + 2px);
  width: calc(1rem - 4px);
  height: calc(1rem - 4px);
  background-color: #adb5bd;
  border-radius: 0.5rem;
  transition: transform 0.15s ease-in-out, background-color 0.15s ease-in-out, border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
}
@media (prefers-reduced-motion: reduce) {
  .custom-switch .custom-control-label::after {
    transition: none;
  }
}
.custom-switch .custom-control-input:checked ~ .custom-control-label::after {
  background-color: #fff;
  transform: translateX(0.75rem);
}
.custom-switch .custom-control-input:disabled:checked ~ .custom-control-label::before {
  background-color: rgba(254, 209, 54, 0.5);
}

.custom-select {
  display: inline-block;
  width: 100%;
  height: calc(1.5em + 0.75rem + 2px);
  padding: 0.375rem 1.75rem 0.375rem 0.75rem;
  font-size: 1rem;
  font-weight: 400;
  line-height: 1.5;
  color: #495057;
  vertical-align: middle;
  background: #fff url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' width='4' height='5' viewBox='0 0 4 5'%3e%3cpath fill='%23343a40' d='M2 0L0 2h4zm0 5L0 3h4z'/%3e%3c/svg%3e") no-repeat right 0.75rem center/8px 10px;
  border: 1px solid #ced4da;
  border-radius: 0.25rem;
  -webkit-appearance: none;
     -moz-appearance: none;
          appearance: none;
}
.custom-select:focus {
  border-color: #ffeeb5;
  outline: 0;
  box-shadow: 0 0 0 0.2rem rgba(254, 209, 54, 0.25);
}
.custom-select:focus::-ms-value {
  color: #495057;
  background-color: #fff;
}
.custom-select[multiple], .custom-select[size]:not([size="1"]) {
  height: auto;
  padding-right: 0.75rem;
  background-image: none;
}
.custom-select:disabled {
  color: #6c757d;
  background-color: #e9ecef;
}
.custom-select::-ms-expand {
  display: none;
}
.custom-select:-moz-focusring {
  color: transparent;
  text-shadow: 0 0 0 #495057;
}

.custom-select-sm {
  height: calc(1.5em + 0.5rem + 2px);
  padding-top: 0.25rem;
  padding-bottom: 0.25rem;
  padding-left: 0.5rem;
  font-size: 0.875rem;
}

.custom-select-lg {
  height: calc(1.5em + 1rem + 2px);
  padding-top: 0.5rem;
  padding-bottom: 0.5rem;
  padding-left: 1rem;
  font-size: 1.25rem;
}

.custom-file {
  position: relative;
  display: inline-block;
  width: 100%;
  height: calc(1.5em + 0.75rem + 2px);
  margin-bottom: 0;
}

.custom-file-input {
  position: relative;
  z-index: 2;
  width: 100%;
  height: calc(1.5em + 0.75rem + 2px);
  margin: 0;
  opacity: 0;
}
.custom-file-input:focus ~ .custom-file-label {
  border-color: #ffeeb5;
  box-shadow: 0 0 0 0.2rem rgba(254, 209, 54, 0.25);
}
.custom-file-input[disabled] ~ .custom-file-label, .custom-file-input:disabled ~ .custom-file-label {
  background-color: #e9ecef;
}
.custom-file-input:lang(en) ~ .custom-file-label::after {
  content: "Browse";
}
.custom-file-input ~ .custom-file-label[data-browse]::after {
  content: attr(data-browse);
}

.custom-file-label {
  position: absolute;
  top: 0;
  right: 0;
  left: 0;
  z-index: 1;
  height: calc(1.5em + 0.75rem + 2px);
  padding: 0.375rem 0.75rem;
  font-weight: 400;
  line-height: 1.5;
  color: #495057;
  background-color: #fff;
  border: 1px solid #ced4da;
  border-radius: 0.25rem;
}
.custom-file-label::after {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  z-index: 3;
  display: block;
  height: calc(1.5em + 0.75rem);
  padding: 0.375rem 0.75rem;
  line-height: 1.5;
  color: #495057;
  content: "Browse";
  background-color: #e9ecef;
  border-left: inherit;
  border-radius: 0 0.25rem 0.25rem 0;
}

.custom-range {
  width: 100%;
  height: 1.4rem;
  padding: 0;
  background-color: transparent;
  -webkit-appearance: none;
     -moz-appearance: none;
          appearance: none;
}
.custom-range:focus {
  outline: none;
}
.custom-range:focus::-webkit-slider-thumb {
  box-shadow: 0 0 0 1px #fff, 0 0 0 0.2rem rgba(254, 209, 54, 0.25);
}
.custom-range:focus::-moz-range-thumb {
  box-shadow: 0 0 0 1px #fff, 0 0 0 0.2rem rgba(254, 209, 54, 0.25);
}
.custom-range:focus::-ms-thumb {
  box-shadow: 0 0 0 1px #fff, 0 0 0 0.2rem rgba(254, 209, 54, 0.25);
}
.custom-range::-moz-focus-outer {
  border: 0;
}
.custom-range::-webkit-slider-thumb {
  width: 1rem;
  height: 1rem;
  margin-top: -0.25rem;
  background-color: #fed136;
  border: 0;
  border-radius: 1rem;
  -webkit-transition: background-color 0.15s ease-in-out, border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
  transition: background-color 0.15s ease-in-out, border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
  -webkit-appearance: none;
          appearance: none;
}
@media (prefers-reduced-motion: reduce) {
  .custom-range::-webkit-slider-thumb {
    -webkit-transition: none;
    transition: none;
  }
}
.custom-range::-webkit-slider-thumb:active {
  background-color: #fffae8;
}
.custom-range::-webkit-slider-runnable-track {
  width: 100%;
  height: 0.5rem;
  color: transparent;
  cursor: pointer;
  background-color: #dee2e6;
  border-color: transparent;
  border-radius: 1rem;
}
.custom-range::-moz-range-thumb {
  width: 1rem;
  height: 1rem;
  background-color: #fed136;
  border: 0;
  border-radius: 1rem;
  -moz-transition: background-color 0.15s ease-in-out, border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
  transition: background-color 0.15s ease-in-out, border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
  -moz-appearance: none;
       appearance: none;
}
@media (prefers-reduced-motion: reduce) {
  .custom-range::-moz-range-thumb {
    -moz-transition: none;
    transition: none;
  }
}
.custom-range::-moz-range-thumb:active {
  background-color: #fffae8;
}
.custom-range::-moz-range-track {
  width: 100%;
  height: 0.5rem;
  color: transparent;
  cursor: pointer;
  background-color: #dee2e6;
  border-color: transparent;
  border-radius: 1rem;
}
.custom-range::-ms-thumb {
  width: 1rem;
  height: 1rem;
  margin-top: 0;
  margin-right: 0.2rem;
  margin-left: 0.2rem;
  background-color: #fed136;
  border: 0;
  border-radius: 1rem;
  -ms-transition: background-color 0.15s ease-in-out, border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
  transition: background-color 0.15s ease-in-out, border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
  appearance: none;
}
@media (prefers-reduced-motion: reduce) {
  .custom-range::-ms-thumb {
    -ms-transition: none;
    transition: none;
  }
}
.custom-range::-ms-thumb:active {
  background-color: #fffae8;
}
.custom-range::-ms-track {
  width: 100%;
  height: 0.5rem;
  color: transparent;
  cursor: pointer;
  background-color: transparent;
  border-color: transparent;
  border-width: 0.5rem;
}
.custom-range::-ms-fill-lower {
  background-color: #dee2e6;
  border-radius: 1rem;
}
.custom-range::-ms-fill-upper {
  margin-right: 15px;
  background-color: #dee2e6;
  border-radius: 1rem;
}
.custom-range:disabled::-webkit-slider-thumb {
  background-color: #adb5bd;
}
.custom-range:disabled::-webkit-slider-runnable-track {
  cursor: default;
}
.custom-range:disabled::-moz-range-thumb {
  background-color: #adb5bd;
}
.custom-range:disabled::-moz-range-track {
  cursor: default;
}
.custom-range:disabled::-ms-thumb {
  background-color: #adb5bd;
}

.custom-control-label::before,
.custom-file-label,
.custom-select {
  transition: background-color 0.15s ease-in-out, border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
}
@media (prefers-reduced-motion: reduce) {
  .custom-control-label::before,
.custom-file-label,
.custom-select {
    transition: none;
  }
}

.nav {
  display: flex;
  flex-wrap: wrap;
  padding-left: 0;
  margin-bottom: 0;
  list-style: none;
}

.nav-link {
  display: block;
  padding: 0.5rem 1rem;
}
.nav-link:hover, .nav-link:focus {
  text-decoration: none;
}
.nav-link.disabled {
  color: #6c757d;
  pointer-events: none;
  cursor: default;
}

.nav-tabs {
  border-bottom: 1px solid #dee2e6;
}
.nav-tabs .nav-item {
  margin-bottom: -1px;
}
.nav-tabs .nav-link {
  border: 1px solid transparent;
  border-top-left-radius: 0.25rem;
  border-top-right-radius: 0.25rem;
}
.nav-tabs .nav-link:hover, .nav-tabs .nav-link:focus {
  border-color: #e9ecef #e9ecef #dee2e6;
}
.nav-tabs .nav-link.disabled {
  color: #6c757d;
  background-color: transparent;
  border-color: transparent;
}
.nav-tabs .nav-link.active,
.nav-tabs .nav-item.show .nav-link {
  color: #495057;
  background-color: #fff;
  border-color: #dee2e6 #dee2e6 #fff;
}
.nav-tabs .dropdown-menu {
  margin-top: -1px;
  border-top-left-radius: 0;
  border-top-right-radius: 0;
}

.nav-pills .nav-link {
  border-radius: 0.25rem;
}
.nav-pills .nav-link.active,
.nav-pills .show > .nav-link {
  color: #fff;
  background-color: #fed136;
}

.nav-fill > .nav-link,
.nav-fill .nav-item {
  flex: 1 1 auto;
  text-align: center;
}

.nav-justified > .nav-link,
.nav-justified .nav-item {
  flex-basis: 0;
  flex-grow: 1;
  text-align: center;
}

.tab-content > .tab-pane {
  display: none;
}
.tab-content > .active {
  display: block;
}

.breadcrumb {
  display: flex;
  flex-wrap: wrap;
  padding: 0.75rem 1rem;
  margin-bottom: 1rem;
  list-style: none;
  background-color: #e9ecef;
  border-radius: 0.25rem;
}

.breadcrumb-item {
  display: flex;
}
.breadcrumb-item + .breadcrumb-item {
  padding-left: 0.5rem;
}
.breadcrumb-item + .breadcrumb-item::before {
  display: inline-block;
  padding-right: 0.5rem;
  color: #6c757d;
  content: "/";
}
.breadcrumb-item + .breadcrumb-item:hover::before {
  text-decoration: underline;
}
.breadcrumb-item + .breadcrumb-item:hover::before {
  text-decoration: none;
}
.breadcrumb-item.active {
  color: #6c757d;
}

.page-item:first-child .page-link {
  margin-left: 0;
  border-top-left-radius: 0.25rem;
  border-bottom-left-radius: 0.25rem;
}
.page-item:last-child .page-link {
  border-top-right-radius: 0.25rem;
  border-bottom-right-radius: 0.25rem;
}
.page-item.active .page-link {
  z-index: 3;
  color: #fff;
  background-color: #fed136;
  border-color: #fed136;
}
.page-item.disabled .page-link {
  color: #6c757d;
  pointer-events: none;
  cursor: auto;
  background-color: #fff;
  border-color: #dee2e6;
}

.pagination-lg .page-link {
  padding: 0.75rem 1.5rem;
  font-size: 1.25rem;
  line-height: 1.5;
}
.pagination-lg .page-item:first-child .page-link {
  border-top-left-radius: 0.3rem;
  border-bottom-left-radius: 0.3rem;
}
.pagination-lg .page-item:last-child .page-link {
  border-top-right-radius: 0.3rem;
  border-bottom-right-radius: 0.3rem;
}

.pagination-sm .page-link {
  padding: 0.25rem 0.5rem;
  font-size: 0.875rem;
  line-height: 1.5;
}
.pagination-sm .page-item:first-child .page-link {
  border-top-left-radius: 0.2rem;
  border-bottom-left-radius: 0.2rem;
}
.pagination-sm .page-item:last-child .page-link {
  border-top-right-radius: 0.2rem;
  border-bottom-right-radius: 0.2rem;
}

.jumbotron {
  padding: 2rem 1rem;
  margin-bottom: 2rem;
  background-color: #e9ecef;
  border-radius: 0.3rem;
}
@media (min-width: 576px) {
  .jumbotron {
    padding: 4rem 2rem;
  }
}

.jumbotron-fluid {
  padding-right: 0;
  padding-left: 0;
  border-radius: 0;
}

@-webkit-keyframes progress-bar-stripes {
  from {
    background-position: 1rem 0;
  }
  to {
    background-position: 0 0;
  }
}

@keyframes progress-bar-stripes {
  from {
    background-position: 1rem 0;
  }
  to {
    background-position: 0 0;
  }
}
.progress {
  display: flex;
  height: 1rem;
  overflow: hidden;
  line-height: 0;
  font-size: 0.75rem;
  background-color: #e9ecef;
  border-radius: 0.25rem;
}

.progress-bar {
  display: flex;
  flex-direction: column;
  justify-content: center;
  overflow: hidden;
  color: #fff;
  text-align: center;
  white-space: nowrap;
  background-color: #fed136;
  transition: width 0.6s ease;
}
@media (prefers-reduced-motion: reduce) {
  .progress-bar {
    transition: none;
  }
}

.progress-bar-striped {
  background-image: linear-gradient(45deg, rgba(255, 255, 255, 0.15) 25%, transparent 25%, transparent 50%, rgba(255, 255, 255, 0.15) 50%, rgba(255, 255, 255, 0.15) 75%, transparent 75%, transparent);
  background-size: 1rem 1rem;
}

.progress-bar-animated {
  -webkit-animation: progress-bar-stripes 1s linear infinite;
          animation: progress-bar-stripes 1s linear infinite;
}
@media (prefers-reduced-motion: reduce) {
  .progress-bar-animated {
    -webkit-animation: none;
            animation: none;
  }
}

.media {
  display: flex;
  align-items: flex-start;
}

.media-body {
  flex: 1;
}

.list-group {
  display: flex;
  flex-direction: column;
  padding-left: 0;
  margin-bottom: 0;
  border-radius: 0.25rem;
}

.list-group-item-action {
  width: 100%;
  color: #495057;
  text-align: inherit;
}
.list-group-item-action:hover, .list-group-item-action:focus {
  z-index: 1;
  color: #495057;
  text-decoration: none;
  background-color: #f8f9fa;
}
.list-group-item-action:active {
  color: #212529;
  background-color: #e9ecef;
}

.list-group-item {
  position: relative;
  display: block;
  padding: 0.75rem 1.25rem;
  background-color: #fff;
  border: 1px solid rgba(0, 0, 0, 0.125);
}
.list-group-item:first-child {
  border-top-left-radius: inherit;
  border-top-right-radius: inherit;
}
.list-group-item:last-child {
  border-bottom-right-radius: inherit;
  border-bottom-left-radius: inherit;
}
.list-group-item.disabled, .list-group-item:disabled {
  color: #6c757d;
  pointer-events: none;
  background-color: #fff;
}
.list-group-item.active {
  z-index: 2;
  color: #fff;
  background-color: #fed136;
  border-color: #fed136;
}
.list-group-item + .list-group-item {
  border-top-width: 0;
}
.list-group-item + .list-group-item.active {
  margin-top: -1px;
  border-top-width: 1px;
}

.list-group-horizontal {
  flex-direction: row;
}
.list-group-horizontal > .list-group-item:first-child {
  border-bottom-left-radius: 0.25rem;
  border-top-right-radius: 0;
}
.list-group-horizontal > .list-group-item:last-child {
  border-top-right-radius: 0.25rem;
  border-bottom-left-radius: 0;
}
.list-group-horizontal > .list-group-item.active {
  margin-top: 0;
}
.list-group-horizontal > .list-group-item + .list-group-item {
  border-top-width: 1px;
  border-left-width: 0;
}
.list-group-horizontal > .list-group-item + .list-group-item.active {
  margin-left: -1px;
  border-left-width: 1px;
}

@media (min-width: 576px) {
  .list-group-horizontal-sm {
    flex-direction: row;
  }
  .list-group-horizontal-sm > .list-group-item:first-child {
    border-bottom-left-radius: 0.25rem;
    border-top-right-radius: 0;
  }
  .list-group-horizontal-sm > .list-group-item:last-child {
    border-top-right-radius: 0.25rem;
    border-bottom-left-radius: 0;
  }
  .list-group-horizontal-sm > .list-group-item.active {
    margin-top: 0;
  }
  .list-group-horizontal-sm > .list-group-item + .list-group-item {
    border-top-width: 1px;
    border-left-width: 0;
  }
  .list-group-horizontal-sm > .list-group-item + .list-group-item.active {
    margin-left: -1px;
    border-left-width: 1px;
  }
}
@media (min-width: 768px) {
  .list-group-horizontal-md {
    flex-direction: row;
  }
  .list-group-horizontal-md > .list-group-item:first-child {
    border-bottom-left-radius: 0.25rem;
    border-top-right-radius: 0;
  }
  .list-group-horizontal-md > .list-group-item:last-child {
    border-top-right-radius: 0.25rem;
    border-bottom-left-radius: 0;
  }
  .list-group-horizontal-md > .list-group-item.active {
    margin-top: 0;
  }
  .list-group-horizontal-md > .list-group-item + .list-group-item {
    border-top-width: 1px;
    border-left-width: 0;
  }
  .list-group-horizontal-md > .list-group-item + .list-group-item.active {
    margin-left: -1px;
    border-left-width: 1px;
  }
}
@media (min-width: 992px) {
  .list-group-horizontal-lg {
    flex-direction: row;
  }
  .list-group-horizontal-lg > .list-group-item:first-child {
    border-bottom-left-radius: 0.25rem;
    border-top-right-radius: 0;
  }
  .list-group-horizontal-lg > .list-group-item:last-child {
    border-top-right-radius: 0.25rem;
    border-bottom-left-radius: 0;
  }
  .list-group-horizontal-lg > .list-group-item.active {
    margin-top: 0;
  }
  .list-group-horizontal-lg > .list-group-item + .list-group-item {
    border-top-width: 1px;
    border-left-width: 0;
  }
  .list-group-horizontal-lg > .list-group-item + .list-group-item.active {
    margin-left: -1px;
    border-left-width: 1px;
  }
}
@media (min-width: 1200px) {
  .list-group-horizontal-xl {
    flex-direction: row;
  }
  .list-group-horizontal-xl > .list-group-item:first-child {
    border-bottom-left-radius: 0.25rem;
    border-top-right-radius: 0;
  }
  .list-group-horizontal-xl > .list-group-item:last-child {
    border-top-right-radius: 0.25rem;
    border-bottom-left-radius: 0;
  }
  .list-group-horizontal-xl > .list-group-item.active {
    margin-top: 0;
  }
  .list-group-horizontal-xl > .list-group-item + .list-group-item {
    border-top-width: 1px;
    border-left-width: 0;
  }
  .list-group-horizontal-xl > .list-group-item + .list-group-item.active {
    margin-left: -1px;
    border-left-width: 1px;
  }
}
.list-group-flush {
  border-radius: 0;
}
.list-group-flush > .list-group-item {
  border-width: 0 0 1px;
}
.list-group-flush > .list-group-item:last-child {
  border-bottom-width: 0;
}

.list-group-item-primary {
  color: #846d1c;
  background-color: #fff2c7;
}
.list-group-item-primary.list-group-item-action:hover, .list-group-item-primary.list-group-item-action:focus {
  color: #846d1c;
  background-color: #ffecae;
}
.list-group-item-primary.list-group-item-action.active {
  color: #fff;
  background-color: #846d1c;
  border-color: #846d1c;
}

.list-group-item-secondary {
  color: #383d41;
  background-color: #d6d8db;
}
.list-group-item-secondary.list-group-item-action:hover, .list-group-item-secondary.list-group-item-action:focus {
  color: #383d41;
  background-color: #c8cbcf;
}
.list-group-item-secondary.list-group-item-action.active {
  color: #fff;
  background-color: #383d41;
  border-color: #383d41;
}

.list-group-item-success {
  color: #155724;
  background-color: #c3e6cb;
}
.list-group-item-success.list-group-item-action:hover, .list-group-item-success.list-group-item-action:focus {
  color: #155724;
  background-color: #b1dfbb;
}
.list-group-item-success.list-group-item-action.active {
  color: #fff;
  background-color: #155724;
  border-color: #155724;
}

.list-group-item-info {
  color: #0c5460;
  background-color: #bee5eb;
}
.list-group-item-info.list-group-item-action:hover, .list-group-item-info.list-group-item-action:focus {
  color: #0c5460;
  background-color: #abdde5;
}
.list-group-item-info.list-group-item-action.active {
  color: #fff;
  background-color: #0c5460;
  border-color: #0c5460;
}

.list-group-item-warning {
  color: #846d1c;
  background-color: #fff2c7;
}
.list-group-item-warning.list-group-item-action:hover, .list-group-item-warning.list-group-item-action:focus {
  color: #846d1c;
  background-color: #ffecae;
}
.list-group-item-warning.list-group-item-action.active {
  color: #fff;
  background-color: #846d1c;
  border-color: #846d1c;
}

.list-group-item-danger {
  color: #721c24;
  background-color: #f5c6cb;
}
.list-group-item-danger.list-group-item-action:hover, .list-group-item-danger.list-group-item-action:focus {
  color: #721c24;
  background-color: #f1b0b7;
}
.list-group-item-danger.list-group-item-action.active {
  color: #fff;
  background-color: #721c24;
  border-color: #721c24;
}

.list-group-item-light {
  color: #818182;
  background-color: #fdfdfe;
}
.list-group-item-light.list-group-item-action:hover, .list-group-item-light.list-group-item-action:focus {
  color: #818182;
  background-color: #ececf6;
}
.list-group-item-light.list-group-item-action.active {
  color: #fff;
  background-color: #818182;
  border-color: #818182;
}

.list-group-item-dark {
  color: #1b1e21;
  background-color: #c6c8ca;
}
.list-group-item-dark.list-group-item-action:hover, .list-group-item-dark.list-group-item-action:focus {
  color: #1b1e21;
  background-color: #b9bbbe;
}
.list-group-item-dark.list-group-item-action.active {
  color: #fff;
  background-color: #1b1e21;
  border-color: #1b1e21;
}

.close {
  float: right;
  font-size: 1.5rem;
  font-weight: 700;
  line-height: 1;
  color: #000;
  text-shadow: 0 1px 0 #fff;
  opacity: 0.5;
}
.close:hover {
  color: #000;
  text-decoration: none;
}
.close:not(:disabled):not(.disabled):hover, .close:not(:disabled):not(.disabled):focus {
  opacity: 0.75;
}

button.close {
  padding: 0;
  background-color: transparent;
  border: 0;
}

a.close.disabled {
  pointer-events: none;
}

.toast {
  flex-basis: 350px;
  max-width: 350px;
  font-size: 0.875rem;
  background-color: rgba(255, 255, 255, 0.85);
  background-clip: padding-box;
  border: 1px solid rgba(0, 0, 0, 0.1);
  box-shadow: 0 0.25rem 0.75rem rgba(0, 0, 0, 0.1);
  opacity: 0;
  border-radius: 0.25rem;
}
.toast:not(:last-child) {
  margin-bottom: 0.75rem;
}
.toast.showing {
  opacity: 1;
}
.toast.show {
  display: block;
  opacity: 1;
}
.toast.hide {
  display: none;
}

.toast-header {
  display: flex;
  align-items: center;
  padding: 0.25rem 0.75rem;
  color: #6c757d;
  background-color: rgba(255, 255, 255, 0.85);
  background-clip: padding-box;
  border-bottom: 1px solid rgba(0, 0, 0, 0.05);
  border-top-left-radius: calc(0.25rem - 1px);
  border-top-right-radius: calc(0.25rem - 1px);
}

.toast-body {
  padding: 0.75rem;
}

.modal-open {
  overflow: hidden;
}
.modal-open .modal {
  overflow-x: hidden;
  overflow-y: auto;
}

.modal {
  position: fixed;
  top: 0;
  left: 0;
  z-index: 1050;
  display: none;
  width: 100%;
  height: 100%;
  overflow: hidden;
  outline: 0;
}

.modal-dialog {
  position: relative;
  width: auto;
  margin: 0.5rem;
  pointer-events: none;
}
.modal.fade .modal-dialog {
  transition: transform 0.3s ease-out;
  transform: translate(0, -50px);
}
@media (prefers-reduced-motion: reduce) {
  .modal.fade .modal-dialog {
    transition: none;
  }
}
.modal.show .modal-dialog {
  transform: none;
}
.modal.modal-static .modal-dialog {
  transform: scale(1.02);
}

.modal-dialog-scrollable {
  display: flex;
  max-height: calc(100% - 1rem);
}
.modal-dialog-scrollable .modal-content {
  max-height: calc(100vh - 1rem);
  overflow: hidden;
}
.modal-dialog-scrollable .modal-header,
.modal-dialog-scrollable .modal-footer {
  flex-shrink: 0;
}
.modal-dialog-scrollable .modal-body {
  overflow-y: auto;
}

.modal-dialog-centered {
  display: flex;
  align-items: center;
  min-height: calc(100% - 1rem);
}
.modal-dialog-centered::before {
  display: block;
  height: calc(100vh - 1rem);
  height: -webkit-min-content;
  height: -moz-min-content;
  height: min-content;
  content: "";
}
.modal-dialog-centered.modal-dialog-scrollable {
  flex-direction: column;
  justify-content: center;
  height: 100%;
}
.modal-dialog-centered.modal-dialog-scrollable .modal-content {
  max-height: none;
}
.modal-dialog-centered.modal-dialog-scrollable::before {
  content: none;
}

.modal-content {
  position: relative;
  display: flex;
  flex-direction: column;
  width: 100%;
  pointer-events: auto;
  background-color: #fff;
  background-clip: padding-box;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 0.3rem;
  outline: 0;
}

.modal-backdrop {
  position: fixed;
  top: 0;
  left: 0;
  z-index: 1040;
  width: 100vw;
  height: 100vh;
  background-color: #000;
}
.modal-backdrop.fade {
  opacity: 0;
}
.modal-backdrop.show {
  opacity: 0.5;
}

.modal-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  padding: 1rem 1rem;
  border-bottom: 1px solid #dee2e6;
  border-top-left-radius: calc(0.3rem - 1px);
  border-top-right-radius: calc(0.3rem - 1px);
}
.modal-header .close {
  padding: 1rem 1rem;
  margin: -1rem -1rem -1rem auto;
}

.modal-title {
  margin-bottom: 0;
  line-height: 1.5;
}

.modal-body {
  position: relative;
  flex: 1 1 auto;
  padding: 1rem;
}

.modal-footer {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: flex-end;
  padding: 0.75rem;
  border-top: 1px solid #dee2e6;
  border-bottom-right-radius: calc(0.3rem - 1px);
  border-bottom-left-radius: calc(0.3rem - 1px);
}
.modal-footer > * {
  margin: 0.25rem;
}

.modal-scrollbar-measure {
  position: absolute;
  top: -9999px;
  width: 50px;
  height: 50px;
  overflow: scroll;
}

@media (min-width: 576px) {
  .modal-dialog {
    max-width: 500px;
    margin: 1.75rem auto;
  }

  .modal-dialog-scrollable {
    max-height: calc(100% - 3.5rem);
  }
  .modal-dialog-scrollable .modal-content {
    max-height: calc(100vh - 3.5rem);
  }

  .modal-dialog-centered {
    min-height: calc(100% - 3.5rem);
  }
  .modal-dialog-centered::before {
    height: calc(100vh - 3.5rem);
    height: -webkit-min-content;
    height: -moz-min-content;
    height: min-content;
  }

  .modal-sm {
    max-width: 300px;
  }
}
@media (min-width: 992px) {
  .modal-lg,
.modal-xl {
    max-width: 800px;
  }
}
@media (min-width: 1200px) {
  .modal-xl {
    max-width: 1140px;
  }
}
.tooltip {
  position: absolute;
  z-index: 1070;
  display: block;
  margin: 0;
  font-family: "Roboto Slab", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji";
  font-style: normal;
  font-weight: 400;
  line-height: 1.5;
  text-align: left;
  text-align: start;
  text-decoration: none;
  text-shadow: none;
  text-transform: none;
  letter-spacing: normal;
  word-break: normal;
  word-spacing: normal;
  white-space: normal;
  line-break: auto;
  font-size: 0.875rem;
  word-wrap: break-word;
  opacity: 0;
}
.tooltip.show {
  opacity: 0.9;
}
.tooltip .arrow {
  position: absolute;
  display: block;
  width: 0.8rem;
  height: 0.4rem;
}
.tooltip .arrow::before {
  position: absolute;
  content: "";
  border-color: transparent;
  border-style: solid;
}

.bs-tooltip-top, .bs-tooltip-auto[x-placement^=top] {
  padding: 0.4rem 0;
}
.bs-tooltip-top .arrow, .bs-tooltip-auto[x-placement^=top] .arrow {
  bottom: 0;
}
.bs-tooltip-top .arrow::before, .bs-tooltip-auto[x-placement^=top] .arrow::before {
  top: 0;
  border-width: 0.4rem 0.4rem 0;
  border-top-color: #000;
}

.bs-tooltip-right, .bs-tooltip-auto[x-placement^=right] {
  padding: 0 0.4rem;
}
.bs-tooltip-right .arrow, .bs-tooltip-auto[x-placement^=right] .arrow {
  left: 0;
  width: 0.4rem;
  height: 0.8rem;
}
.bs-tooltip-right .arrow::before, .bs-tooltip-auto[x-placement^=right] .arrow::before {
  right: 0;
  border-width: 0.4rem 0.4rem 0.4rem 0;
  border-right-color: #000;
}

.bs-tooltip-bottom, .bs-tooltip-auto[x-placement^=bottom] {
  padding: 0.4rem 0;
}
.bs-tooltip-bottom .arrow, .bs-tooltip-auto[x-placement^=bottom] .arrow {
  top: 0;
}
.bs-tooltip-bottom .arrow::before, .bs-tooltip-auto[x-placement^=bottom] .arrow::before {
  bottom: 0;
  border-width: 0 0.4rem 0.4rem;
  border-bottom-color: #000;
}

.bs-tooltip-left, .bs-tooltip-auto[x-placement^=left] {
  padding: 0 0.4rem;
}
.bs-tooltip-left .arrow, .bs-tooltip-auto[x-placement^=left] .arrow {
  right: 0;
  width: 0.4rem;
  height: 0.8rem;
}
.bs-tooltip-left .arrow::before, .bs-tooltip-auto[x-placement^=left] .arrow::before {
  left: 0;
  border-width: 0.4rem 0 0.4rem 0.4rem;
  border-left-color: #000;
}

.tooltip-inner {
  max-width: 200px;
  padding: 0.25rem 0.5rem;
  color: #fff;
  text-align: center;
  background-color: #000;
  border-radius: 0.25rem;
}

.popover {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 1060;
  display: block;
  max-width: 276px;
  font-family: "Roboto Slab", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji";
  font-style: normal;
  font-weight: 400;
  line-height: 1.5;
  text-align: left;
  text-align: start;
  text-decoration: none;
  text-shadow: none;
  text-transform: none;
  letter-spacing: normal;
  word-break: normal;
  word-spacing: normal;
  white-space: normal;
  line-break: auto;
  font-size: 0.875rem;
  word-wrap: break-word;
  background-color: #fff;
  background-clip: padding-box;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 0.3rem;
}
.popover .arrow {
  position: absolute;
  display: block;
  width: 1rem;
  height: 0.5rem;
  margin: 0 0.3rem;
}
.popover .arrow::before, .popover .arrow::after {
  position: absolute;
  display: block;
  content: "";
  border-color: transparent;
  border-style: solid;
}

.bs-popover-top, .bs-popover-auto[x-placement^=top] {
  margin-bottom: 0.5rem;
}
.bs-popover-top > .arrow, .bs-popover-auto[x-placement^=top] > .arrow {
  bottom: calc(-0.5rem - 1px);
}
.bs-popover-top > .arrow::before, .bs-popover-auto[x-placement^=top] > .arrow::before {
  bottom: 0;
  border-width: 0.5rem 0.5rem 0;
  border-top-color: rgba(0, 0, 0, 0.25);
}
.bs-popover-top > .arrow::after, .bs-popover-auto[x-placement^=top] > .arrow::after {
  bottom: 1px;
  border-width: 0.5rem 0.5rem 0;
  border-top-color: #fff;
}

.bs-popover-right, .bs-popover-auto[x-placement^=right] {
  margin-left: 0.5rem;
}
.bs-popover-right > .arrow, .bs-popover-auto[x-placement^=right] > .arrow {
  left: calc(-0.5rem - 1px);
  width: 0.5rem;
  height: 1rem;
  margin: 0.3rem 0;
}
.bs-popover-right > .arrow::before, .bs-popover-auto[x-placement^=right] > .arrow::before {
  left: 0;
  border-width: 0.5rem 0.5rem 0.5rem 0;
  border-right-color: rgba(0, 0, 0, 0.25);
}
.bs-popover-right > .arrow::after, .bs-popover-auto[x-placement^=right] > .arrow::after {
  left: 1px;
  border-width: 0.5rem 0.5rem 0.5rem 0;
  border-right-color: #fff;
}

.bs-popover-bottom, .bs-popover-auto[x-placement^=bottom] {
  margin-top: 0.5rem;
}
.bs-popover-bottom > .arrow, .bs-popover-auto[x-placement^=bottom] > .arrow {
  top: calc(-0.5rem - 1px);
}
.bs-popover-bottom > .arrow::before, .bs-popover-auto[x-placement^=bottom] > .arrow::before {
  top: 0;
  border-width: 0 0.5rem 0.5rem 0.5rem;
  border-bottom-color: rgba(0, 0, 0, 0.25);
}
.bs-popover-bottom > .arrow::after, .bs-popover-auto[x-placement^=bottom] > .arrow::after {
  top: 1px;
  border-width: 0 0.5rem 0.5rem 0.5rem;
  border-bottom-color: #fff;
}
.bs-popover-bottom .popover-header::before, .bs-popover-auto[x-placement^=bottom] .popover-header::before {
  position: absolute;
  top: 0;
  left: 50%;
  display: block;
  width: 1rem;
  margin-left: -0.5rem;
  content: "";
  border-bottom: 1px solid #f7f7f7;
}

.bs-popover-left, .bs-popover-auto[x-placement^=left] {
  margin-right: 0.5rem;
}
.bs-popover-left > .arrow, .bs-popover-auto[x-placement^=left] > .arrow {
  right: calc(-0.5rem - 1px);
  width: 0.5rem;
  height: 1rem;
  margin: 0.3rem 0;
}
.bs-popover-left > .arrow::before, .bs-popover-auto[x-placement^=left] > .arrow::before {
  right: 0;
  border-width: 0.5rem 0 0.5rem 0.5rem;
  border-left-color: rgba(0, 0, 0, 0.25);
}
.bs-popover-left > .arrow::after, .bs-popover-auto[x-placement^=left] > .arrow::after {
  right: 1px;
  border-width: 0.5rem 0 0.5rem 0.5rem;
  border-left-color: #fff;
}

.popover-header {
  padding: 0.5rem 0.75rem;
  margin-bottom: 0;
  font-size: 1rem;
  background-color: #f7f7f7;
  border-bottom: 1px solid #ebebeb;
  border-top-left-radius: calc(0.3rem - 1px);
  border-top-right-radius: calc(0.3rem - 1px);
}
.popover-header:empty {
  display: none;
}

.popover-body {
  padding: 0.5rem 0.75rem;
  color: #212529;
}

.carousel {
  position: relative;
  height: 400px;
}

.carousel.pointer-event {
  touch-action: pan-y;
}

.carousel-inner {
  /*position: relative;*/
  width: 100%;
  /*overflow: hidden;*/
}
.carousel-inner::after {
  display: block;
  clear: both;
  content: "";
}

.carousel-item {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  left: 0;
  /*position: relative;*/
  display: none;
  width: 60%;
  height: auto;
  margin: 0 auto;
  -webkit-backface-visibility: hidden;
          backface-visibility: hidden;
  transition: transform 1.0s ease-in-out;
}
@media (prefers-reduced-motion: reduce) {
  .carousel-item {
    transition: none;
  }
}

.carousel-item.active,
.carousel-item-next,
.carousel-item-prev {
  display: block;
}

.carousel-item-next:not(.carousel-item-left),
.active.carousel-item-right {
  transform: translateX(100%);
}

.carousel-item-prev:not(.carousel-item-right),
.active.carousel-item-left {
  transform: translateX(-100%);
}

.carousel-fade .carousel-item {
  opacity: 0;
  transition-property: opacity;
  transform: none;
}
.carousel-fade .carousel-item.active,
.carousel-fade .carousel-item-next.carousel-item-left,
.carousel-fade .carousel-item-prev.carousel-item-right {
  z-index: 1;
  opacity: 1;
}
.carousel-fade .active.carousel-item-left,
.carousel-fade .active.carousel-item-right {
  z-index: 0;
  opacity: 0;
  transition: opacity 0s 0.6s;
}
@media (prefers-reduced-motion: reduce) {
  .carousel-fade .active.carousel-item-left,
.carousel-fade .active.carousel-item-right {
    transition: none;
  }
}

.carousel-control-prev,
.carousel-control-next {
  position: absolute;
  top: 0;
  bottom: 0;
  z-index: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 15%;
  color: #fff;
  text-align: center;
  opacity: 0.5;
  transition: opacity 0.15s ease;
}
@media (prefers-reduced-motion: reduce) {
  .carousel-control-prev,
.carousel-control-next {
    transition: none;
  }
}
.carousel-control-prev:hover, .carousel-control-prev:focus,
.carousel-control-next:hover,
.carousel-control-next:focus {
  color: #fff;
  text-decoration: none;
  outline: 0;
  opacity: 0.9;
}

.carousel-control-prev {
  left: 0;
}

.carousel-control-next {
  right: 0;
}

.carousel-control-prev-icon,
.carousel-control-next-icon {
  display: inline-block;
  width: 40px;
  height: 40px;
  background: no-repeat 50%/100% 100%;
}

.carousel-control-prev-icon {
  background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='%23000' width='8' height='8' viewBox='0 0 8 8'%3e%3cpath d='M5.25 0l-4 4 4 4 1.5-1.5L4.25 4l2.5-2.5L5.25 0z'/%3e%3c/svg%3e");
}

.carousel-control-next-icon {
  background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='%23000' width='8' height='8' viewBox='0 0 8 8'%3e%3cpath d='M2.75 0l-1.5 1.5L3.75 4l-2.5 2.5L2.75 8l4-4-4-4z'/%3e%3c/svg%3e");
}

.carousel-indicators {
  position: absolute;
  right: 0;
  bottom: 0;
  left: 0;
  z-index: 15;
  display: flex;
  justify-content: center;
  padding-left: 0;
  margin-right: 15%;
  margin-left: 15%;
  list-style: none;
}
.carousel-indicators li {
  box-sizing: content-box;
  flex: 0 1 auto;
  width: 30px;
  height: 3px;
  margin-right: 3px;
  margin-left: 3px;
  text-indent: -999px;
  cursor: pointer;
  background-color: #B20000;
  background-clip: padding-box;
  border-top: 10px solid transparent;
  border-bottom: 10px solid transparent;
  opacity: 0.25;
  transition: opacity 0.6s ease;
}
@media (prefers-reduced-motion: reduce) {
  .carousel-indicators li {
    transition: none;
  }
}
.carousel-indicators .active {
  opacity: 1;
}

.carousel-caption {
  position: absolute;
  right: 15%;
  bottom: 20px;
  left: 15%;
  z-index: 10;
  padding-top: 20px;
  padding-bottom: 20px;
  color: #fff;
  text-align: center;
}

@-webkit-keyframes spinner-border {
  to {
    transform: rotate(360deg);
  }
}

@keyframes spinner-border {
  to {
    transform: rotate(360deg);
  }
}
.spinner-border {
  display: inline-block;
  width: 2rem;
  height: 2rem;
  vertical-align: text-bottom;
  border: 0.25em solid currentColor;
  border-right-color: transparent;
  border-radius: 50%;
  -webkit-animation: spinner-border 0.75s linear infinite;
          animation: spinner-border 0.75s linear infinite;
}

.spinner-border-sm {
  width: 1rem;
  height: 1rem;
  border-width: 0.2em;
}

@-webkit-keyframes spinner-grow {
  0% {
    transform: scale(0);
  }
  50% {
    opacity: 1;
    transform: none;
  }
}

@keyframes spinner-grow {
  0% {
    transform: scale(0);
  }
  50% {
    opacity: 1;
    transform: none;
  }
}
.spinner-grow {
  display: inline-block;
  width: 2rem;
  height: 2rem;
  vertical-align: text-bottom;
  background-color: currentColor;
  border-radius: 50%;
  opacity: 0;
  -webkit-animation: spinner-grow 0.75s linear infinite;
          animation: spinner-grow 0.75s linear infinite;
}

.spinner-grow-sm {
  width: 1rem;
  height: 1rem;
}

.img-paper {
  margin-top: 1rem;
  height: 100px;
}

.align-baseline {
  vertical-align: baseline !important;
}

.align-top {
  vertical-align: top !important;
}

.align-middle {
  vertical-align: middle !important;
}

.align-bottom {
  vertical-align: bottom !important;
}

.align-text-bottom {
  vertical-align: text-bottom !important;
}

.align-text-top {
  vertical-align: text-top !important;
}

.bg-primary {
  background-color: #fed136 !important;
}

a.bg-primary:hover, a.bg-primary:focus,
button.bg-primary:hover,
button.bg-primary:focus {
  background-color: #fec503 !important;
}

.bg-secondary {
  background-color: #6c757d !important;
}

a.bg-secondary:hover, a.bg-secondary:focus,
button.bg-secondary:hover,
button.bg-secondary:focus {
  background-color: #545b62 !important;
}

.bg-success {
  background-color: #28a745 !important;
}

a.bg-success:hover, a.bg-success:focus,
button.bg-success:hover,
button.bg-success:focus {
  background-color: #1e7e34 !important;
}

.bg-info {
  background-color: #17a2b8 !important;
}

a.bg-info:hover, a.bg-info:focus,
button.bg-info:hover,
button.bg-info:focus {
  background-color: #117a8b !important;
}

.bg-warning {
  background-color: #fed136 !important;
}

a.bg-warning:hover, a.bg-warning:focus,
button.bg-warning:hover,
button.bg-warning:focus {
  background-color: #fec503 !important;
}

.bg-danger {
  background-color: #dc3545 !important;
}

a.bg-danger:hover, a.bg-danger:focus,
button.bg-danger:hover,
button.bg-danger:focus {
  background-color: #bd2130 !important;
}

.bg-light {
  background-color: #f8f9fa !important;
}

a.bg-light:hover, a.bg-light:focus,
button.bg-light:hover,
button.bg-light:focus {
  background-color: #dae0e5 !important;
}

.bg-dark {
  background-color: #343a40 !important;
}

a.bg-dark:hover, a.bg-dark:focus,
button.bg-dark:hover,
button.bg-dark:focus {
  background-color: #1d2124 !important;
}

.bg-white {
  background-color: #fff !important;
}

.bg-transparent {
  background-color: transparent !important;
}

.border {
  border: 1px solid #dee2e6 !important;
}

.border-top {
  border-top: 1px solid #dee2e6 !important;
}

.border-right {
  border-right: 1px solid #dee2e6 !important;
}

.border-bottom {
  border-bottom: 1px solid #dee2e6 !important;
}

.border-left {
  border-left: 1px solid #dee2e6 !important;
}

.border-0 {
  border: 0 !important;
}

.border-top-0 {
  border-top: 0 !important;
}

.border-right-0 {
  border-right: 0 !important;
}

.border-bottom-0 {
  border-bottom: 0 !important;
}

.border-left-0 {
  border-left: 0 !important;
}

.border-primary {
  border-color: #fed136 !important;
}

.border-secondary {
  border-color: #6c757d !important;
}

.border-success {
  border-color: #28a745 !important;
}

.border-info {
  border-color: #17a2b8 !important;
}

.border-warning {
  border-color: #fed136 !important;
}

.border-danger {
  border-color: #dc3545 !important;
}

.border-light {
  border-color: #f8f9fa !important;
}

.border-dark {
  border-color: #343a40 !important;
}

.border-white {
  border-color: #fff !important;
}

.rounded-sm {
  border-radius: 0.2rem !important;
}

.rounded {
  border-radius: 0.25rem !important;
}

.rounded-top {
  border-top-left-radius: 0.25rem !important;
  border-top-right-radius: 0.25rem !important;
}

.rounded-right {
  border-top-right-radius: 0.25rem !important;
  border-bottom-right-radius: 0.25rem !important;
}

.rounded-bottom {
  border-bottom-right-radius: 0.25rem !important;
  border-bottom-left-radius: 0.25rem !important;
}

.rounded-left {
  border-top-left-radius: 0.25rem !important;
  border-bottom-left-radius: 0.25rem !important;
}

.rounded-lg {
  border-radius: 0.3rem !important;
}

.rounded-circle {
  border-radius: 50% !important;
}

.rounded-pill {
  border-radius: 50rem !important;
}

.rounded-0 {
  border-radius: 0 !important;
}

.clearfix::after {
  display: block;
  clear: both;
  content: "";
}

.d-none {
  display: none !important;
}

.d-inline {
  display: inline !important;
}

.d-inline-block {
  display: inline-block !important;
}

.d-block {
  display: block !important;
}

.d-table {
  display: table !important;
}

.d-table-row {
  display: table-row !important;
}

.d-table-cell {
  display: table-cell !important;
}

.d-flex {
  display: flex !important;
}

.d-inline-flex {
  display: inline-flex !important;
}

@media (min-width: 576px) {
  .d-sm-none {
    display: none !important;
  }

  .d-sm-inline {
    display: inline !important;
  }

  .d-sm-inline-block {
    display: inline-block !important;
  }

  .d-sm-block {
    display: block !important;
  }

  .d-sm-table {
    display: table !important;
  }

  .d-sm-table-row {
    display: table-row !important;
  }

  .d-sm-table-cell {
    display: table-cell !important;
  }

  .d-sm-flex {
    display: flex !important;
  }

  .d-sm-inline-flex {
    display: inline-flex !important;
  }
}
@media (min-width: 768px) {
  .d-md-none {
    display: none !important;
  }

  .d-md-inline {
    display: inline !important;
  }

  .d-md-inline-block {
    display: inline-block !important;
  }

  .d-md-block {
    display: block !important;
  }

  .d-md-table {
    display: table !important;
  }

  .d-md-table-row {
    display: table-row !important;
  }

  .d-md-table-cell {
    display: table-cell !important;
  }

  .d-md-flex {
    display: flex !important;
  }

  .d-md-inline-flex {
    display: inline-flex !important;
  }
}
@media (min-width: 992px) {
  .d-lg-none {
    display: none !important;
  }

  .d-lg-inline {
    display: inline !important;
  }

  .d-lg-inline-block {
    display: inline-block !important;
  }

  .d-lg-block {
    display: block !important;
  }

  .d-lg-table {
    display: table !important;
  }

  .d-lg-table-row {
    display: table-row !important;
  }

  .d-lg-table-cell {
    display: table-cell !important;
  }

  .d-lg-flex {
    display: flex !important;
  }

  .d-lg-inline-flex {
    display: inline-flex !important;
  }
}
@media (min-width: 1200px) {
  .d-xl-none {
    display: none !important;
  }

  .d-xl-inline {
    display: inline !important;
  }

  .d-xl-inline-block {
    display: inline-block !important;
  }

  .d-xl-block {
    display: block !important;
  }

  .d-xl-table {
    display: table !important;
  }

  .d-xl-table-row {
    display: table-row !important;
  }

  .d-xl-table-cell {
    display: table-cell !important;
  }

  .d-xl-flex {
    display: flex !important;
  }

  .d-xl-inline-flex {
    display: inline-flex !important;
  }
}
@media print {
  .d-print-none {
    display: none !important;
  }

  .d-print-inline {
    display: inline !important;
  }

  .d-print-inline-block {
    display: inline-block !important;
  }

  .d-print-block {
    display: block !important;
  }

  .d-print-table {
    display: table !important;
  }

  .d-print-table-row {
    display: table-row !important;
  }

  .d-print-table-cell {
    display: table-cell !important;
  }

  .d-print-flex {
    display: flex !important;
  }

  .d-print-inline-flex {
    display: inline-flex !important;
  }
}
.embed-responsive {
  position: relative;
  display: block;
  width: 100%;
  padding: 0;
  overflow: hidden;
}
.embed-responsive::before {
  display: block;
  content: "";
}
.embed-responsive .embed-responsive-item,
.embed-responsive iframe,
.embed-responsive embed,
.embed-responsive object,
.embed-responsive video {
  position: absolute;
  top: 0;
  bottom: 0;
  left: 0;
  width: 100%;
  height: 100%;
  border: 0;
}

.embed-responsive-21by9::before {
  padding-top: 42.8571428571%;
}

.embed-responsive-16by9::before {
  padding-top: 56.25%;
}

.embed-responsive-4by3::before {
  padding-top: 75%;
}

.embed-responsive-1by1::before {
  padding-top: 100%;
}

.flex-row {
  flex-direction: row !important;
}

.flex-column {
  flex-direction: column !important;
}

.flex-row-reverse {
  flex-direction: row-reverse !important;
}

.flex-column-reverse {
  flex-direction: column-reverse !important;
}

.flex-wrap {
  flex-wrap: wrap !important;
}

.flex-nowrap {
  flex-wrap: nowrap !important;
}

.flex-wrap-reverse {
  flex-wrap: wrap-reverse !important;
}

.flex-fill {
  flex: 1 1 auto !important;
}

.flex-grow-0 {
  flex-grow: 0 !important;
}

.flex-grow-1 {
  flex-grow: 1 !important;
}

.flex-shrink-0 {
  flex-shrink: 0 !important;
}

.flex-shrink-1 {
  flex-shrink: 1 !important;
}

.justify-content-start {
  justify-content: flex-start !important;
}

.justify-content-end {
  justify-content: flex-end !important;
}

.justify-content-center {
  justify-content: center !important;
}

.justify-content-between {
  justify-content: space-between !important;
}

.justify-content-around {
  justify-content: space-around !important;
}

.align-items-start {
  align-items: flex-start !important;
}

.align-items-end {
  align-items: flex-end !important;
}

.align-items-center {
  align-items: center !important;
}

.align-items-baseline {
  align-items: baseline !important;
}

.align-items-stretch {
  align-items: stretch !important;
}

.align-content-start {
  align-content: flex-start !important;
}

.align-content-end {
  align-content: flex-end !important;
}

.align-content-center {
  align-content: center !important;
}

.align-content-between {
  align-content: space-between !important;
}

.align-content-around {
  align-content: space-around !important;
}

.align-content-stretch {
  align-content: stretch !important;
}

.align-self-auto {
  align-self: auto !important;
}

.align-self-start {
  align-self: flex-start !important;
}

.align-self-end {
  align-self: flex-end !important;
}

.align-self-center {
  align-self: center !important;
}

.align-self-baseline {
  align-self: baseline !important;
}

.align-self-stretch {
  align-self: stretch !important;
}

@media (min-width: 576px) {
  .flex-sm-row {
    flex-direction: row !important;
  }

  .flex-sm-column {
    flex-direction: column !important;
  }

  .flex-sm-row-reverse {
    flex-direction: row-reverse !important;
  }

  .flex-sm-column-reverse {
    flex-direction: column-reverse !important;
  }

  .flex-sm-wrap {
    flex-wrap: wrap !important;
  }

  .flex-sm-nowrap {
    flex-wrap: nowrap !important;
  }

  .flex-sm-wrap-reverse {
    flex-wrap: wrap-reverse !important;
  }

  .flex-sm-fill {
    flex: 1 1 auto !important;
  }

  .flex-sm-grow-0 {
    flex-grow: 0 !important;
  }

  .flex-sm-grow-1 {
    flex-grow: 1 !important;
  }

  .flex-sm-shrink-0 {
    flex-shrink: 0 !important;
  }

  .flex-sm-shrink-1 {
    flex-shrink: 1 !important;
  }

  .justify-content-sm-start {
    justify-content: flex-start !important;
  }

  .justify-content-sm-end {
    justify-content: flex-end !important;
  }

  .justify-content-sm-center {
    justify-content: center !important;
  }

  .justify-content-sm-between {
    justify-content: space-between !important;
  }

  .justify-content-sm-around {
    justify-content: space-around !important;
  }

  .align-items-sm-start {
    align-items: flex-start !important;
  }

  .align-items-sm-end {
    align-items: flex-end !important;
  }

  .align-items-sm-center {
    align-items: center !important;
  }

  .align-items-sm-baseline {
    align-items: baseline !important;
  }

  .align-items-sm-stretch {
    align-items: stretch !important;
  }

  .align-content-sm-start {
    align-content: flex-start !important;
  }

  .align-content-sm-end {
    align-content: flex-end !important;
  }

  .align-content-sm-center {
    align-content: center !important;
  }

  .align-content-sm-between {
    align-content: space-between !important;
  }

  .align-content-sm-around {
    align-content: space-around !important;
  }

  .align-content-sm-stretch {
    align-content: stretch !important;
  }

  .align-self-sm-auto {
    align-self: auto !important;
  }

  .align-self-sm-start {
    align-self: flex-start !important;
  }

  .align-self-sm-end {
    align-self: flex-end !important;
  }

  .align-self-sm-center {
    align-self: center !important;
  }

  .align-self-sm-baseline {
    align-self: baseline !important;
  }

  .align-self-sm-stretch {
    align-self: stretch !important;
  }
}
@media (min-width: 768px) {
  .flex-md-row {
    flex-direction: row !important;
  }

  .flex-md-column {
    flex-direction: column !important;
  }

  .flex-md-row-reverse {
    flex-direction: row-reverse !important;
  }

  .flex-md-column-reverse {
    flex-direction: column-reverse !important;
  }

  .flex-md-wrap {
    flex-wrap: wrap !important;
  }

  .flex-md-nowrap {
    flex-wrap: nowrap !important;
  }

  .flex-md-wrap-reverse {
    flex-wrap: wrap-reverse !important;
  }

  .flex-md-fill {
    flex: 1 1 auto !important;
  }

  .flex-md-grow-0 {
    flex-grow: 0 !important;
  }

  .flex-md-grow-1 {
    flex-grow: 1 !important;
  }

  .flex-md-shrink-0 {
    flex-shrink: 0 !important;
  }

  .flex-md-shrink-1 {
    flex-shrink: 1 !important;
  }

  .justify-content-md-start {
    justify-content: flex-start !important;
  }

  .justify-content-md-end {
    justify-content: flex-end !important;
  }

  .justify-content-md-center {
    justify-content: center !important;
  }

  .justify-content-md-between {
    justify-content: space-between !important;
  }

  .justify-content-md-around {
    justify-content: space-around !important;
  }

  .align-items-md-start {
    align-items: flex-start !important;
  }

  .align-items-md-end {
    align-items: flex-end !important;
  }

  .align-items-md-center {
    align-items: center !important;
  }

  .align-items-md-baseline {
    align-items: baseline !important;
  }

  .align-items-md-stretch {
    align-items: stretch !important;
  }

  .align-content-md-start {
    align-content: flex-start !important;
  }

  .align-content-md-end {
    align-content: flex-end !important;
  }

  .align-content-md-center {
    align-content: center !important;
  }

  .align-content-md-between {
    align-content: space-between !important;
  }

  .align-content-md-around {
    align-content: space-around !important;
  }

  .align-content-md-stretch {
    align-content: stretch !important;
  }

  .align-self-md-auto {
    align-self: auto !important;
  }

  .align-self-md-start {
    align-self: flex-start !important;
  }

  .align-self-md-end {
    align-self: flex-end !important;
  }

  .align-self-md-center {
    align-self: center !important;
  }

  .align-self-md-baseline {
    align-self: baseline !important;
  }

  .align-self-md-stretch {
    align-self: stretch !important;
  }
}
@media (min-width: 992px) {
  .flex-lg-row {
    flex-direction: row !important;
  }

  .flex-lg-column {
    flex-direction: column !important;
  }

  .flex-lg-row-reverse {
    flex-direction: row-reverse !important;
  }

  .flex-lg-column-reverse {
    flex-direction: column-reverse !important;
  }

  .flex-lg-wrap {
    flex-wrap: wrap !important;
  }

  .flex-lg-nowrap {
    flex-wrap: nowrap !important;
  }

  .flex-lg-wrap-reverse {
    flex-wrap: wrap-reverse !important;
  }

  .flex-lg-fill {
    flex: 1 1 auto !important;
  }

  .flex-lg-grow-0 {
    flex-grow: 0 !important;
  }

  .flex-lg-grow-1 {
    flex-grow: 1 !important;
  }

  .flex-lg-shrink-0 {
    flex-shrink: 0 !important;
  }

  .flex-lg-shrink-1 {
    flex-shrink: 1 !important;
  }

  .justify-content-lg-start {
    justify-content: flex-start !important;
  }

  .justify-content-lg-end {
    justify-content: flex-end !important;
  }

  .justify-content-lg-center {
    justify-content: center !important;
  }

  .justify-content-lg-between {
    justify-content: space-between !important;
  }

  .justify-content-lg-around {
    justify-content: space-around !important;
  }

  .align-items-lg-start {
    align-items: flex-start !important;
  }

  .align-items-lg-end {
    align-items: flex-end !important;
  }

  .align-items-lg-center {
    align-items: center !important;
  }

  .align-items-lg-baseline {
    align-items: baseline !important;
  }

  .align-items-lg-stretch {
    align-items: stretch !important;
  }

  .align-content-lg-start {
    align-content: flex-start !important;
  }

  .align-content-lg-end {
    align-content: flex-end !important;
  }

  .align-content-lg-center {
    align-content: center !important;
  }

  .align-content-lg-between {
    align-content: space-between !important;
  }

  .align-content-lg-around {
    align-content: space-around !important;
  }

  .align-content-lg-stretch {
    align-content: stretch !important;
  }

  .align-self-lg-auto {
    align-self: auto !important;
  }

  .align-self-lg-start {
    align-self: flex-start !important;
  }

  .align-self-lg-end {
    align-self: flex-end !important;
  }

  .align-self-lg-center {
    align-self: center !important;
  }

  .align-self-lg-baseline {
    align-self: baseline !important;
  }

  .align-self-lg-stretch {
    align-self: stretch !important;
  }
}
@media (min-width: 1200px) {
  .flex-xl-row {
    flex-direction: row !important;
  }

  .flex-xl-column {
    flex-direction: column !important;
  }

  .flex-xl-row-reverse {
    flex-direction: row-reverse !important;
  }

  .flex-xl-column-reverse {
    flex-direction: column-reverse !important;
  }

  .flex-xl-wrap {
    flex-wrap: wrap !important;
  }

  .flex-xl-nowrap {
    flex-wrap: nowrap !important;
  }

  .flex-xl-wrap-reverse {
    flex-wrap: wrap-reverse !important;
  }

  .flex-xl-fill {
    flex: 1 1 auto !important;
  }

  .flex-xl-grow-0 {
    flex-grow: 0 !important;
  }

  .flex-xl-grow-1 {
    flex-grow: 1 !important;
  }

  .flex-xl-shrink-0 {
    flex-shrink: 0 !important;
  }

  .flex-xl-shrink-1 {
    flex-shrink: 1 !important;
  }

  .justify-content-xl-start {
    justify-content: flex-start !important;
  }

  .justify-content-xl-end {
    justify-content: flex-end !important;
  }

  .justify-content-xl-center {
    justify-content: center !important;
  }

  .justify-content-xl-between {
    justify-content: space-between !important;
  }

  .justify-content-xl-around {
    justify-content: space-around !important;
  }

  .align-items-xl-start {
    align-items: flex-start !important;
  }

  .align-items-xl-end {
    align-items: flex-end !important;
  }

  .align-items-xl-center {
    align-items: center !important;
  }

  .align-items-xl-baseline {
    align-items: baseline !important;
  }

  .align-items-xl-stretch {
    align-items: stretch !important;
  }

  .align-content-xl-start {
    align-content: flex-start !important;
  }

  .align-content-xl-end {
    align-content: flex-end !important;
  }

  .align-content-xl-center {
    align-content: center !important;
  }

  .align-content-xl-between {
    align-content: space-between !important;
  }

  .align-content-xl-around {
    align-content: space-around !important;
  }

  .align-content-xl-stretch {
    align-content: stretch !important;
  }

  .align-self-xl-auto {
    align-self: auto !important;
  }

  .align-self-xl-start {
    align-self: flex-start !important;
  }

  .align-self-xl-end {
    align-self: flex-end !important;
  }

  .align-self-xl-center {
    align-self: center !important;
  }

  .align-self-xl-baseline {
    align-self: baseline !important;
  }

  .align-self-xl-stretch {
    align-self: stretch !important;
  }
}
.float-left {
  float: left !important;
}

.float-right {
  float: right !important;
}

.float-none {
  float: none !important;
}

@media (min-width: 576px) {
  .float-sm-left {
    float: left !important;
  }

  .float-sm-right {
    float: right !important;
  }

  .float-sm-none {
    float: none !important;
  }
}
@media (min-width: 768px) {
  .float-md-left {
    float: left !important;
  }

  .float-md-right {
    float: right !important;
  }

  .float-md-none {
    float: none !important;
  }
}
@media (min-width: 992px) {
  .float-lg-left {
    float: left !important;
  }

  .float-lg-right {
    float: right !important;
  }

  .float-lg-none {
    float: none !important;
  }
}
@media (min-width: 1200px) {
  .float-xl-left {
    float: left !important;
  }

  .float-xl-right {
    float: right !important;
  }

  .float-xl-none {
    float: none !important;
  }
}

.overflow-auto {
  overflow: auto !important;
}

.overflow-hidden {
  overflow: hidden !important;
}

@media print {
  *,
*::before,
*::after {
    text-shadow: none !important;
    box-shadow: none !important;
  }

  a:not(.btn) {
    text-decoration: underline;
  }

  abbr[title]::after {
    content: " (" attr(title) ")";
  }

  pre {
    white-space: pre-wrap !important;
  }

  pre,
blockquote {
    border: 1px solid #adb5bd;
    page-break-inside: avoid;
  }

  thead {
    display: table-header-group;
  }

  tr,
img {
    page-break-inside: avoid;
  }

  p,
h2,
h3 {
    orphans: 3;
    widows: 3;
  }

  h2,
h3 {
    page-break-after: avoid;
  }

  @page {
    size: a3;
  }
  body {
    min-width: 992px !important;
  }

  .container {
    min-width: 992px !important;
  }

  .table-bordered th,
.table-bordered td {
    border: 1px solid #dee2e6 !important;
  }

  .table-dark {
    color: inherit;
  }
  .table-dark th,
.table-dark td,
.table-dark thead th,
.table-dark tbody + tbody {
    border-color: #dee2e6;
  }

  .table .thead-dark th {
    color: inherit;
    border-color: #dee2e6;
  }
}
body {
  overflow-x: hidden;
}

p {
  line-height: 1.75;
}

.page-section {
  padding: 6rem 0;
}
.row-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
}

.row-grid .cell {
  background: #faf0f0;
  margin: 10px;
  padding: 20px;
}

</style>


<div class container>
  <div class="row text-center">
    <div class="col-lg-3 col-md-6">
        <div class="feature-box">
            <h5 class="my-3">6-DoF Intuitive User Interface</h5>
            <p class="text-muted">We provide 6 degree of freedom intuitive motion control which maps phone movement to robot arm movement.</p>
        </div>
    </div>
    <div class="col-lg-3 col-md-6">
        <div class="feature-box">
            <h5 class="my-3">Simultaneous Users</h5>
            <p class="text-muted">RoboTurk can host multiple simultaenous users that each control a robot arm in its own workspace, as well as multiple users that control robot arms in a shared workspace, allowing for demonstrations on collaborative and adversarial tasks.</p>
        </div>
    </div>
    <div class="col-lg-3 col-md-6">
        <div class="feature-box">
            <h5 class="my-3">Worldwide Low-Latency Robot Teleoperation</h5>
            <p class="text-muted">Real-time robot control of simulated and physical robot arms from across the world. This has been stress-tested by controlling robot arms at Stanford from far locations such as China and India.</p>
        </div>
    </div>
    <div class="col-lg-3 col-md-6">
        <div class="feature-box">
            <h5 class="my-3">Human-in-the-Loop Intervention Mechanism</h5>
            <p class="text-muted">Users can watch an autonomous robot arm try to solve tasks and provide assistance when necessary, helping the robot learn from its mistakes.</p>
        </div>
    </div>
  </div>
</div>


<!-- <style>
  .column {
      width: 45%;
      float: left;
      margin-right: 3%;
      margin-bottom: 20px;
      text-align: center;
      padding: 20px;
      height: 250px;
  }

  .column:last-child {
      margin-right: 0;
  }

  .clear {
      clear: both;
  }
  
  @media screen and (max-width : 1024px) {
    .column {
		width: 50%;
		float: left;
		margin-right: 0;
		padding: 15px;
	}
}

@media screen and (max-width : 767px) {
    .column {
		width: 100%;
		float: none;
		padding: 15px 0;
	}
}
</style>

<div class="row">
  <div class="column" style="background-color:#DEEEFB;">
    <h4>Suite of Learning Algorithms</h4>
    <p>High-quality implementations of offline learning algorithms, including BC, BC-RNN, HBC, IRIS, BCQ, CQL, and TD3-BC</p>
  </div>
  <div class="column" style="background-color:#DEEEFB;">
    <h4>Standardized Datasets</h4>
    <p>Datasets collected from different sources (single proficient human, multiple humans, and machine-generated) across simulated and real-world tasks spanning multiple robots and environments</p>
  </div>
  <div class="column" style="background-color:#DEEEFB;">
    <h4>Modular Design</h4>
    <p>Support for learning both low-dimensional and visuomotor policies, diverse network architectures, and interface to easily use external datasets</p>
  </div>
  <div class="column" style="background-color:#DEEEFB;">
    <h4>Flexible Experiment Workflow</h4>
    <p>Utilities for running hyperparameter sweeps, visualizing demonstration data and trained policies, and collecting new datasets using trained policies</p>
  </div>
</div>
-->


<!-- 1. **Offline Learning Algorithms**
High-quality implementations of offline learning algorithms, including BC, BC-RNN, HBC, IRIS, BCQ, CQL, and TD3-BC
2. **Standardized Datasets**
Datasets collected from different sources (single proficient human, multiple humans, and machine-generated) across simulated and real-world tasks spanning multiple robots and environments
3. **Modular Design**
Support for learning both low-dimensional and visuomotor policies, diverse network architectures, and interface to easily use external datasets
4. **Flexible Experiment Workflow**
Utilities for running hyperparameter sweeps, visualizing demonstration data and trained policies, and collecting new datasets using trained policies -->

## Reproducing benchmark study results

The robomimic framework also makes reproducing the results from this [benchmark study](https://arise-initiative.github.io/robomimic-web/study) easy. See the [reproducing results documentation](./results.html) for more information.

## Troubleshooting

Please see the [troubleshooting](../miscellaneous/troubleshooting.html) section for common fixes, or [submit an issue](https://github.com/ARISE-Initiative/robomimic/issues) on our github page.

## Contributing to robomimic
This project is part of the broader [Advancing Robot Intelligence through Simulated Environments (ARISE) Initiative](https://github.com/ARISE-Initiative), with the aim of lowering the barriers of entry for cutting-edge research at the intersection of AI and Robotics.
The project originally began development in late 2018 by researchers in the [Stanford Vision and Learning Lab](http://svl.stanford.edu/) (SVL).
Now it is actively maintained and used for robotics research projects across multiple labs.
We welcome community contributions to this project.
For details please check our [contributing guidelines](../miscellaneous/contributing.html).

## Citation

Please cite [this paper](https://arxiv.org/abs/2108.03298) if you use this framework in your work:

```
@inproceedings{robomimic2021,
  title={What Matters in Learning from Offline Human Demonstrations for Robot Manipulation},
  author={Ajay Mandlekar and Danfei Xu and Josiah Wong and Soroush Nasiriany and Chen Wang and Rohun Kulkarni and Li Fei-Fei and Silvio Savarese and Yuke Zhu and Roberto Mart\'{i}n-Mart\'{i}n},
  booktitle={Conference on Robot Learning (CoRL)},
  year={2021}
}
```